import argparse
from minireranker import MiniReranker

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--image", default=None)
parser.add_argument("--video_frames", nargs="*", default=None)
parser.add_argument("--video_root", default=None)
parser.add_argument("--prune_k", type=float, default=None)
args = parser.parse_args()
model = MiniReranker(args.model, video_root=args.video_root, prune_k=args.prune_k)
query = {"text": "Find content about a dog playing outdoors."}
documents = [{"text": "A dog plays in a park."}, {"text": "An indoor cooking lesson."}]
if args.image:
    documents = [{"image": args.image}]
if args.video_frames:
    documents = [{"video": args.video_frames}]
print(model.process({"query": query, "documents": documents}))
