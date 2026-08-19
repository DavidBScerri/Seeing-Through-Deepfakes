"""
Generative-AI detection subsystem.

Groups the evidence streams that decide whether an image is AI-generated
at all — metadata/provenance heuristics, the visual classifier, the
decision-fusion engine, and (stubs for later work) watermark, hash and
robustness evaluation. The downstream deepfake stage lives outside this
package, at ``src.deepfake_detection``, because it runs conditionally on a
fused-positive verdict.
"""
