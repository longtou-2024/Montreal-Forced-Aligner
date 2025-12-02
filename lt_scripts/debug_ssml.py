from endpoints.utils import parse_ssml
#from montreal_forced_aligner.endpoints.utils import parse_ssml


ssml = """
 <speak>아니, 왜 하필 제일 높은 사람한테 돌진한 거냐고!</speak>
"""

ssml_results = parse_ssml(ssml)
breakpoint()
