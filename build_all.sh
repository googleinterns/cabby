# Check the types in everything.
pytype cabby

# Build everything
bazel query cabby/... | xargs bazel build
