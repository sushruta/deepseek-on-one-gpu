import argparse

from inference.engine import InferenceEngine

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", action="store_true", help="Print detailed progress logs")
args = parser.parse_args()

engine = InferenceEngine(
    model_path="/tmp/models/DeepSeek-R1-0528",
    expert_cache_gb=256.0,
    verbose=args.verbose,
)

question = """
what are the top 5 largest cities in the world in terms of population?
"""

answer = engine.generate(question)

print(f"Q: {question}\n\n------------\n\n{answer}\n\n")
