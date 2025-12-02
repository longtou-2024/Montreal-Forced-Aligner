import xml.etree.ElementTree as ET
import string
import io
import copy

from pydub import AudioSegment
import numpy as np

def parse_timepoints(ssml_results, mfa_results):
    """edge case 대응
    1) <mark> 태그로 시작하고 첫 단어 앞에 sil 존재 유무 (대응 완료)
    2) <mark> 태그로 끝나고 마지막 단어 뒤에 sil 존재 유무 (대응 필요 X)
    """
    translator = str.maketrans('', '', string.punctuation)
    mfa_entries = mfa_results['tiers']['words']['entries']
    word_idx = 0
    results = []
    for ssml_item in ssml_results:
        if "text" not in ssml_item: continue

        text_seg = ssml_item["text"]
        text_seg_wo_punc = text_seg.translate(translator)
        tag_name = ssml_item["tag_name"]
        if tag_name == "mark":
            start, end, _word = mfa_entries[word_idx]
            if word_idx == 0 and _word == "<eps>": # case 1)
                word_idx += 1
                start, end, _word = mfa_entries[word_idx]
            mark_name = ssml_item["mark_name"]
            t_point = {
                'markName': mark_name,
                'timeSeconds': start
            }
            results.append(t_point)

        for word in text_seg_wo_punc.split():
            start, end, _word = mfa_entries[word_idx]
            if _word == "<eps>":
                word_idx += 1
                start, end, _word = mfa_entries[word_idx]
            if _word != word:
                raise Exception(f"{word} != {_word}")
            word_idx += 1

    if ssml_results[-1]['tag_name'] == "mark" and ssml_results[-1].get('text') is None: # case 2
        start, end, _word = mfa_entries[-1]
        if _word == "<eps>":
            t_point = {
                'markName': ssml_results[-1]['mark_name'],
                'timeSeconds': start
            }
        else:
            t_point = {
                'markName': ssml_results[-1]['mark_name'],
                'timeSeconds': end
            }

        results.append(t_point)

    return results

def parse_ssml(ssml_string):
    """
    SSML 문자열을 파싱하고, 태그와 텍스트 내용을 추출하는 함수
    """
    avail_tag_set = {"speak", "mark", "prosody"}
    try:
        # SSML 문자열을 ElementTree 루트 요소로 파싱합니다.
        # ET.fromstring()은 SSML의 최상위 태그인 <speak>를 루트로 가정합니다.
        root = ET.fromstring(ssml_string)

        #print(f"루트 태그: {root.tag}")
        #print("-" * 20)

        # SSML 내용을 순회하며 처리합니다.
        results = []
        for i, element in enumerate(root.iter()):
            # 태그 이름 추출 시 네임스페이스 제거 (SSML은 보통 네임스페이스를 사용합니다)
            tag_name = element.tag.split('}')[-1] if '}' in element.tag else element.tag
            tag_name = tag_name.lower()

            if tag_name.lower() not in avail_tag_set:
                raise Exception(f"Invalid tag: {tag_name}")


            item = {"tag_name": tag_name}
            if tag_name == "speak":
                if element.text and element.text.strip():
                    # 요소의 시작 태그와 첫 번째 자식 요소(또는 끝 태그) 사이에 있는 텍스트 콘텐츠입니다.
                    item["text"] = element.text.strip()
            elif tag_name == 'prosody':
                # <prosody> 태그 처리 (속성 추출)
                attrs = {
                    "rate": element.get('rate'),
                    "pitch": element.get('pitch'),
                }
                item["attrs"] = attrs
                if element.text and element.text.strip():
                    item["text"] = element.text.strip()
            elif tag_name == 'mark':
                mark_name = element.get("name")
                item["mark_name"] = mark_name
                if element.tail and element.tail.strip():
                    # 해당 요소의 끝 태그와 바로 다음 형제 요소의 시작 태그 사이에 있는 텍스트입니다.
                    item["text"] = element.tail.strip()
            else:
                 raise Exception(f"Invalid tag: {tag_name}")
            results.append(item)

        # 검증
        # 1) remove text between <speak> and <prosody>
        # <speak></prosody>...
        if len(results) > 1 and results[1]['tag_name'] == 'prosody' and 'text' in results[0]:
            del results[0]['text']
        # 2) unique tag
        if len([item for item in results if item['tag_name'] == 'speak']) > 1:
            raise Exception(f"<speak> tag must be unique")
        if len([item for item in results if item['tag_name'] == 'prosody']) > 1:
            raise Exception(f"<prosody> tag must be unique")

        return results

    except ET.ParseError as e:
        raise Exception(f"SSML 파싱 오류: {e}")


def limit_silence_duration(audio_buf: io.BytesIO, audio_format: str, mfa_result: dict, max_dur_s: float):
    # NOTE(longtou): set minimum
    max_dur_s = np.clip(max_dur_s, 0.1, None).item()

    audio_segment = AudioSegment.from_file(audio_buf, format=audio_format)
    #origin_audio_segment = audio_segment.copy()
    #print(f"Total duration: {len(audio_segment)}ms")

    out_mfa_result = copy.deepcopy(mfa_result)
    out_word_entries = out_mfa_result['tiers']['words']['entries']
    word_entries = mfa_result['tiers']['words']['entries']

    max_dur_ms = max_dur_s * 1000
    trimmed_audio_segments = []
    for i in range(len(word_entries)):
        word_begin_s, word_end_s, word = word_entries[i]
        word_begin_ms = word_begin_s * 1000
        word_end_ms = word_end_s * 1000
        if word == "<eps>" and (word_end_s - word_begin_s) > max_dur_s:
            delta_ms = (word_end_ms - word_begin_ms) - max_dur_ms
            word_middle_ms = (word_end_ms + word_begin_ms) / 2
            word_slice_begin_ms = word_middle_ms - (delta_ms/2)
            word_slice_end_ms = word_middle_ms + (delta_ms/2)
            trim_audio_seg = audio_segment[word_begin_ms:word_slice_begin_ms] + audio_segment[word_slice_end_ms:word_end_ms]
            trimmed_audio_segments.append(trim_audio_seg)

            # fix timestamp
            delta_s = delta_ms / 1000
            out_word_entries[i][1] = out_word_entries[i][1] - delta_s
            for j in range(i+1, len(out_word_entries)):
                out_word_entries[j][0] = out_word_entries[j][0] - delta_s
                out_word_entries[j][1] = out_word_entries[j][1] - delta_s
        else:
            trimmed_audio_segments.append(audio_segment[word_begin_ms:word_end_ms])
    trimmed_audio_segments = sum(trimmed_audio_segments)
    out_audio_buf = io.BytesIO()

    if audio_format == "wav":
        trimmed_audio_segments.export(out_audio_buf, format=audio_format)
    elif audio_format == "mp3":
        trimmed_audio_segments.export(out_audio_buf, format=audio_format, bitrate="128k")
    else:
        raise Exception(f"audio_format: {audio_format} not supproted")
    out_audio_buf.seek(0)

    # NOTE(longtou): set float time precision to 2
    for i in range(len(out_word_entries)):
        out_word_entries[i][0] = round(out_word_entries[i][0], 2)
        # edge case) exclude rounding at last idx & correct 'end' value
        if i == len(out_word_entries) - 1:
            out_mfa_result['end'] = out_word_entries[i][1]
        else:
            out_word_entries[i][1] = round(out_word_entries[i][1], 2)

    return out_audio_buf, out_mfa_result
