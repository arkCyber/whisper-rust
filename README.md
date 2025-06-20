# 🎙️ Enhanced Whisper-RS

[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![License: Unlicense](https://img.shields.io/badge/license-Unlicense-blue.svg)](http://unlicense.org/)
[![GitHub Stars](https://img.shields.io/github/stars/arkCyber/whisper-rust.svg)](https://github.com/arkCyber/whisper-rust/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/arkCyber/whisper-rust.svg)](https://github.com/arkCyber/whisper-rust/network)

Enhanced Rust bindings to [whisper.cpp](https://github.com/ggerganov/whisper.cpp) with comprehensive Chinese language support, advanced error handling, and detailed logging capabilities.

## ✨ Features

### 🔥 **Enhanced Functionality**
- **Multi-language Speech Recognition**: English, Chinese (Simplified/Traditional), and 97+ other languages
- **Advanced Error Handling**: Comprehensive error management with detailed logging
- **Timestamped Logging**: Complete operation tracking with UNIX timestamps
- **Audio Format Validation**: Automatic audio format checking and conversion
- **Chinese Character Detection**: Automatic language detection with visual indicators (🇨🇳/🇺🇸)
- **DTW Token Timestamps**: Dynamic Time Warping for precise token-level timing

### 📋 **Examples Included**
1. **`basic_use.rs`** - Enhanced basic speech-to-text transcription
2. **`audio_transcription.rs`** - Advanced DTW token-level timestamps
3. **`chinese_test.rs`** - Specialized Chinese speech recognition

### 🛠️ **Development Features**
- **138 Unit Tests**: Comprehensive test coverage for all functionality
- **Memory Safety**: Rust's memory safety with C++ whisper.cpp performance
- **Thread Safety**: Safe concurrent processing capabilities
- **Documentation**: Extensive inline documentation in English and Chinese

## 🚀 Quick Start

### Prerequisites

- Rust 1.70+ 
- Git with submodules support
- CMake (for building whisper.cpp)
- A C++ compiler (GCC, Clang, or MSVC)

### Installation

```bash
# Clone the repository with submodules
git clone --recursive https://github.com/arkCyber/whisper-rust.git
cd whisper-rust

# Build the project
cargo build --release
```

### Download Models

```bash
# Download Whisper models
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin

# Download test audio
wget https://github.com/ggml-org/whisper.cpp/raw/master/samples/jfk.wav
```

## 📖 Usage Examples

### Basic English Speech Recognition

```bash
cargo run --example basic_use ggml-small.bin jfk.wav
```

**Output:**
```
[1750407839] INFO: Starting enhanced Whisper transcription example
[1750407839] INFO: Validating command line arguments
[1750407839] INFO: Model path: ggml-small.bin
[1750407839] INFO: WAV path: jfk.wav
...
=== TRANSCRIPTION RESULTS ===
[0 - 400]: And so, my fellow Americans, ask not what your country can do for you,
[400 - 544]: ask what you can do for your country.
=== END TRANSCRIPTION RESULTS ===
[1750407839] INFO: Transcription completed successfully in 1.44 seconds
```

### Chinese Speech Recognition

```bash
cargo run --example chinese_test ggml-small.bin chinese_audio.wav
```

**Output:**
```
[1750407927] INFO: 开始中文语音识别示例 (Starting enhanced Chinese speech recognition example)
...
=== 中文转录结果 (CHINESE TRANSCRIPTION RESULTS) ===
[0 - 250] 🇨🇳: 你好，欢迎使用增强版语音识别系统
[250 - 500] 🇨🇳: 这是一个支持中文的语音转文字示例
=== 中文转录结果结束 (END CHINESE TRANSCRIPTION RESULTS) ===
```

### Programmatic Usage

```rust
use whisper_rs::{WhisperContext, WhisperContextParameters, FullParams, SamplingStrategy};

fn main() -> Result<(), Box<dyn std::error::Error>> {
	// Load model
	let ctx = WhisperContext::new_with_params(
		"ggml-small.bin",
		WhisperContextParameters::default()
	)?;

	// Create transcription parameters
	let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
	params.set_language(Some("zh")); // Chinese
	params.set_print_timestamps(false);

	// Process audio (16kHz, mono, f32)
	let audio_data = load_audio_file("audio.wav")?;
	
	// Run transcription
	let mut state = ctx.create_state()?;
	state.full(params, &audio_data)?;

	// Extract results
	let num_segments = state.full_n_segments()?;
	for i in 0..num_segments {
		let segment = state.full_get_segment_text(i)?;
		let start = state.full_get_segment_t0(i)?;
		let end = state.full_get_segment_t1(i)?;
		println!("[{} - {}]: {}", start, end, segment);
	}

	Ok(())
}
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all 138 unit tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test module
cargo test whisper_state::tests

# Run documentation tests
cargo test --doc
```

**Test Coverage:**
- ✅ Audio processing and conversion
- ✅ Error handling and recovery
- ✅ Memory safety and thread safety
- ✅ Unicode and Chinese character processing
- ✅ Timestamp and token validation
- ✅ Model loading and state management

## 🎯 Supported Languages

| Language | Code | Support Level | Example |
|----------|------|---------------|---------|
| English | `en` | ⭐⭐⭐⭐⭐ | Perfect recognition |
| Chinese (Simplified) | `zh` | ⭐⭐⭐⭐⭐ | 完美支持简体中文 |
| Chinese (Traditional) | `zh` | ⭐⭐⭐⭐⭐ | 完美支援繁體中文 |
| Japanese | `ja` | ⭐⭐⭐⭐ | 日本語対応 |
| Korean | `ko` | ⭐⭐⭐⭐ | 한국어 지원 |
| Spanish | `es` | ⭐⭐⭐⭐ | Soporte en español |
| French | `fr` | ⭐⭐⭐⭐ | Support français |
| German | `de` | ⭐⭐⭐⭐ | Deutsche Unterstützung |
| ... | ... | ... | 97+ languages total |

## 🔧 Feature Flags

Enable additional features during compilation:

```bash
# CUDA GPU acceleration
cargo build --features cuda

# OpenBLAS support  
cargo build --features openblas

# Metal (macOS) acceleration
cargo build --features metal

# Vulkan GPU support
cargo build --features vulkan

# Logging integration
cargo build --features log_backend

# Tracing support
cargo build --features tracing_backend
```

## 📊 Performance Benchmarks

| Model | Size | Languages | Speed (CPU) | Speed (GPU) | Quality |
|-------|------|-----------|-------------|-------------|---------|
| tiny | 39 MB | 99 | ~10x realtime | ~50x realtime | Good |
| small | 244 MB | 99 | ~6x realtime | ~25x realtime | Very Good |
| medium | 769 MB | 99 | ~2x realtime | ~10x realtime | Excellent |
| large | 1550 MB | 99 | ~1x realtime | ~4x realtime | Outstanding |

## 🛠️ Building from Source

### Windows
```bash
# Install Visual Studio Build Tools
# Install CMake
git clone --recursive https://github.com/arkCyber/whisper-rust.git
cd whisper-rust
cargo build --release
```

### macOS
```bash
# Install Xcode Command Line Tools
xcode-select --install
git clone --recursive https://github.com/arkCyber/whisper-rust.git
cd whisper-rust
cargo build --release
```

### Linux
```bash
# Install dependencies
sudo apt-get update
sudo apt-get install build-essential cmake git
git clone --recursive https://github.com/arkCyber/whisper-rust.git
cd whisper-rust
cargo build --release
```

## 🤝 Contributing

We welcome contributions! Please see our contribution guidelines:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### Development Guidelines

- Follow Rust coding conventions
- Add comprehensive tests for new features
- Include both English and Chinese documentation
- Ensure all tests pass: `cargo test`
- Run clippy: `cargo clippy`
- Format code: `cargo fmt`

## 📝 Changelog

### Latest Enhancements

#### 🎉 **v0.14.3-enhanced**
- ✅ **Enhanced Chinese Support**: Specialized Chinese speech recognition
- ✅ **Comprehensive Error Handling**: Full error management system
- ✅ **Timestamped Logging**: Complete operation tracking
- ✅ **Audio Validation**: Format checking and conversion
- ✅ **138 Unit Tests**: Comprehensive test coverage
- ✅ **Multi-language Detection**: Automatic language identification
- ✅ **Enhanced Documentation**: Bilingual code documentation

## 📄 License

This project is in the **public domain** under the [Unlicense](http://unlicense.org/).

```
This is free and unencumbered software released into the public domain.
Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any means.
```

## 🙏 Acknowledgments

- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) - The underlying C++ implementation
- [OpenAI Whisper](https://github.com/openai/whisper) - The original model
- [tazz4843/whisper-rs](https://github.com/tazz4843/whisper-rs) - Original Rust bindings
- All contributors and testers

## 📬 Support

- **Issues**: [GitHub Issues](https://github.com/arkCyber/whisper-rust/issues)
- **Discussions**: [GitHub Discussions](https://github.com/arkCyber/whisper-rust/discussions)
- **Email**: Create an issue for support

---

<div align="center">

**🎙️ Enhanced Whisper-RS - Bringing Advanced Speech Recognition to Rust with Chinese Support! 🚀**

Made with ❤️ by the arkCyber team

</div>
