.PHONY: setup build
setup:
	rustup target add wasm32-unknown-unknown
	cargo install wasm-bindgen-cli
	cargo install wasm-pack

build:
	wasm-pack build --target web --release
