#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

cargo build -p exploreui-backend
cargo build -p exploreui-frontend --target wasm32-unknown-unknown --release

cp target/wasm32-unknown-unknown/release/exploreui_frontend.wasm \
   experiments/exploreui/frontend/dist/app.wasm
cp webui/webui.js \
   experiments/exploreui/frontend/dist/webui.js

echo "Build complete. Run: cargo run -p exploreui-backend"
