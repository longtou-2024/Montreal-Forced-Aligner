# longtou.2024
from __future__ import annotations

from pathlib import Path
from time import time
import argparse

import pywrapfst
import rich_click as click
from kalpy.feat.cmvn import CmvnComputer
from kalpy.fstext.lexicon import HierarchicalCtm, LexiconCompiler
from kalpy.utterance import Segment
from kalpy.utterance import Utterance as KalpyUtterance
from tqdm import tqdm

from montreal_forced_aligner import config
from montreal_forced_aligner.alignment import PretrainedAligner
from montreal_forced_aligner.command_line.utils import (
    common_options,
    validate_acoustic_model,
    validate_dictionary,
    validate_g2p_model,
)
from montreal_forced_aligner.corpus.classes import FileData
from montreal_forced_aligner.data import (
    BRACKETED_WORD,
    CUTOFF_WORD,
    LAUGHTER_WORD,
    OOV_WORD,
    Language,
)
from montreal_forced_aligner.dictionary.mixins import (
    DEFAULT_BRACKETS,
    DEFAULT_CLITIC_MARKERS,
    DEFAULT_COMPOUND_MARKERS,
    DEFAULT_PUNCTUATION,
    DEFAULT_WORD_BREAK_MARKERS,
)
from montreal_forced_aligner.models import AcousticModel, G2PModel
from montreal_forced_aligner.online.alignment import align_utterance_online
from montreal_forced_aligner.tokenization.simple import SimpleTokenizer
from montreal_forced_aligner.tokenization.spacy import generate_language_tokenizer
from montreal_forced_aligner.exceptions import AlignerError


class FakeContext:
    def __init__(self):
        self.args = []
        # set default params
        params = dict()
        # required
        params["sound_file_path"] = ""
        params["text_file_path"] = ""
        params["dictionary_path"] = ""
        params["acoustic_model_path"] = ""
        params["output_path"] = ""

        # options
        params["config_path"] = ""
        params["output_format"] = "json"
        params["no_tokenization"] = False
        params["g2p_model_path"] = ""

        # common options
        params["profile"] = None
        params["temporary_directory"] = config.TEMPORARY_DIRECTORY
        params["num_jobs"] = 1
        # NOTE(longtou): without clean, it reuse previous lexicon compiler
        params["clean"] = True
        params["final_clean"] = True
        params["verbose"] = False
        params["quiet"] = False
        params["overwrite"] = True
        params["use_mp"] = None
        params["use_threading"] = None
        params["debug"] = False
        params["use_postgres"] = False
        params["single_speaker"] = False
        params["cleanup_textgrids"] = None
        self.params = params

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if k in ("sound_file_path", "text_file_path", "dictionary_path", "output_path"):
                v = Path(v)
            self.params[k] = v

def setup_mfa(dict_path="korean_espeak.dict", acoustic_path="korean_espeak.zip", g2p_path=None):
    context = FakeContext()
    context.set_params(
        dictionary_path=dict_path,
        acoustic_model_path=acoustic_path,
        g2p_model_path=g2p_path,
    )
    kwargs = context.params

    if kwargs.get("profile", None) is not None:
        config.profile = kwargs.pop("profile")
    config.update_configuration(kwargs)
    config_path = kwargs.get("config_path", None)

    dictionary_path: Path = kwargs["dictionary_path"]
    acoustic_model_path = kwargs["acoustic_model_path"]
    g2p_model_path = kwargs.get("g2p_model_path", None)
    no_tokenization = kwargs["no_tokenization"]

    acoustic_model = AcousticModel(acoustic_model_path)
    g2p_model = None
    if g2p_model_path:
        g2p_model_path = validate_g2p_model(context, kwargs, g2p_model_path)
        g2p_model = G2PModel(g2p_model_path)
    c = PretrainedAligner.parse_parameters(config_path, context.params, context.args)
    extracted_models_dir = config.TEMPORARY_DIRECTORY.joinpath("extracted_models", "dictionary")
    dictionary_directory = extracted_models_dir.joinpath(dictionary_path.stem)
    dictionary_directory.mkdir(parents=True, exist_ok=True)
    lexicon_compiler = LexiconCompiler(
        disambiguation=False,
        silence_probability=acoustic_model.parameters["silence_probability"],
        initial_silence_probability=acoustic_model.parameters["initial_silence_probability"],
        final_silence_correction=acoustic_model.parameters["final_silence_correction"],
        final_non_silence_correction=acoustic_model.parameters["final_non_silence_correction"],
        silence_phone=acoustic_model.parameters["optional_silence_phone"],
        oov_phone=acoustic_model.parameters["oov_phone"],
        position_dependent_phones=acoustic_model.parameters["position_dependent_phones"],
        phones=acoustic_model.parameters["non_silence_phones"],
        ignore_case=c.get("ignore_case", True),
    )
    l_fst_path = dictionary_directory.joinpath("L.fst")
    l_align_fst_path = dictionary_directory.joinpath("L_align.fst")
    words_path = dictionary_directory.joinpath("words.txt")
    phones_path = dictionary_directory.joinpath("phones.txt")
    if l_fst_path.exists() and not config.CLEAN:
        lexicon_compiler.load_l_from_file(l_fst_path)
        lexicon_compiler.load_l_align_from_file(l_align_fst_path)
        lexicon_compiler.word_table = pywrapfst.SymbolTable.read_text(words_path)
        lexicon_compiler.phone_table = pywrapfst.SymbolTable.read_text(phones_path)
    else:
        lexicon_compiler.load_pronunciations(dictionary_path)
        lexicon_compiler.fst.write(str(l_fst_path))
        lexicon_compiler.align_fst.write(str(l_align_fst_path))
        lexicon_compiler.word_table.write_text(words_path)
        lexicon_compiler.phone_table.write_text(phones_path)
        lexicon_compiler.clear()

    if no_tokenization or acoustic_model.language is Language.unknown:
        tokenizer = SimpleTokenizer(
            word_table=lexicon_compiler.word_table,
            word_break_markers=c.get("word_break_markers", DEFAULT_WORD_BREAK_MARKERS),
            punctuation=c.get("punctuation", DEFAULT_PUNCTUATION),
            clitic_markers=c.get("clitic_markers", DEFAULT_CLITIC_MARKERS),
            compound_markers=c.get("compound_markers", DEFAULT_COMPOUND_MARKERS),
            brackets=c.get("brackets", DEFAULT_BRACKETS),
            laughter_word=c.get("laughter_word", LAUGHTER_WORD),
            oov_word=c.get("oov_word", OOV_WORD),
            bracketed_word=c.get("bracketed_word", BRACKETED_WORD),
            cutoff_word=c.get("cutoff_word", CUTOFF_WORD),
            ignore_case=c.get("ignore_case", True),
        )
    else:
        tokenizer = generate_language_tokenizer(acoustic_model.language)

    return acoustic_model, g2p_model, lexicon_compiler, tokenizer, c


def align_one(
    sound_file_path: Path,
    text_file_path: Path,
    output_path: Path,
    output_format,
    acoustic_model,
    g2p_model,
    lexicon_compiler,
    tokenizer,
    c,
) -> None:
    """
    Align a single file with a pronunciation dictionary and a pretrained acoustic model.
    """
    if output_path.is_dir():
        output_path = output_path.joinpath(sound_file_path.stem + ".TextGrid")

    file_name = sound_file_path.stem
    file = FileData.parse_file(file_name, sound_file_path, text_file_path, "", 0)
    file_ctm = HierarchicalCtm([])
    utterances = []
    cmvn_computer = CmvnComputer()
    for utterance in file.utterances:
        seg = Segment(sound_file_path, utterance.begin, utterance.end, utterance.channel)
        utt = KalpyUtterance(seg, utterance.text)
        utt.generate_mfccs(acoustic_model.mfcc_computer)
        utterances.append(utt)

    cmvn = cmvn_computer.compute_cmvn_from_features([utt.mfccs for utt in utterances])
    align_options = {
        k: v
        for k, v in c.items()
        if k
        in [
            "beam",
            "retry_beam",
            "acoustic_scale",
            "transition_scale",
            "self_loop_scale",
            "boost_silence",
        ]
    }
    for utt in utterances:
        utt.apply_cmvn(cmvn)
        ctm = align_utterance_online(
            acoustic_model,
            utt,
            lexicon_compiler,
            tokenizer=tokenizer,
            g2p_model=g2p_model,
            **align_options,
        )
        file_ctm.word_intervals.extend(ctm.word_intervals)
    if str(output_path) != "-":
        output_path.parent.mkdir(parents=True, exist_ok=True)
    file_ctm.export_textgrid(
        output_path, file_duration=file.wav_info.duration, output_format=output_format
    )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dictionary_path", default="korean_espeak.dict")
    parser.add_argument("acoustic_model_path", default="korean_espeak.zip")
    parser.add_argument("g2p_model_path", default="korean_espeak.zip")
    parser.add_argument("indir", default="tmp")
    parser.add_argument("outdir", default="outdir")
    parser.add_argument("--beam", default=10, type=int)
    parser.add_argument("--retry_beam", default=40, type=int)
    args = parser.parse_args()

    start = time()
    acoustic_model, g2p_model, lexicon_compiler, tokenizer, conf = setup_mfa(args.dictionary_path, args.acoustic_model_path, args.g2p_model_path)
    print(f"setup_mfa: {time()-start}s elapsed")

    # NOTE(longtou): custom option
    conf["beam"] = args.beam
    conf["retry_beam"] = args.retry_beam

    t_arr = []
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    for audio_path in tqdm(Path(args.indir).glob("*.wav")):
        text_path = audio_path.with_suffix(".lab")
        if not text_path.exists():
            text_path = text_path.with_suffix(".txt")
        output_path = Path(f"{outdir}/{text_path.stem}.json")

        start = time()
        try:
            align_one(
                audio_path,
                text_path,
                output_path,
                "json",
                acoustic_model,
                g2p_model,
                lexicon_compiler,
                tokenizer,
                conf,
            )
        except AlignerError as e:
            print(e)
            print(f"{audio_path.stem} align fail")
        t_arr.append(time()-start)

    print(t_arr)
    print(sum(t_arr) / len(t_arr))
