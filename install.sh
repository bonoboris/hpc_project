BASE_DIR=$(dirname $0)
DEFAULT_BUILD_DIR="$BASE_DIR/build"
BUILD_DIR="${1:-$DEFAULT_BUILD_DIR}"

echo "Base dir: $BASE_DIR"
echo "Default build dir: $DEFAULT_BUILD_DIR"
echo "Build dir: $BUILD_DIR"

if [[ ! -d $BUILD_DIR ]]; then
    mkdir $BUILD_DIR
fi

cmake -S $BASE_DIR -B $BUILD_DIR
make -C $BUILD_DIR
