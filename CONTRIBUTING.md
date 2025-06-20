# 🤝 Contributing to Enhanced Whisper-RS

Thank you for your interest in contributing to Enhanced Whisper-RS! We welcome contributions from developers around the world, especially those working on multi-language speech recognition and Chinese language support.

## 🌟 How to Contribute

### 🐛 Reporting Bugs

Before reporting a bug, please:

1. **Search existing issues** to avoid duplicates
2. **Test with the latest version** to ensure the bug still exists
3. **Provide clear reproduction steps**

When reporting a bug, please include:

- **Rust version**: `rustc --version`
- **Operating system and version**
- **Error messages** (full stack traces if available)
- **Minimal reproduction code**
- **Expected vs actual behavior**

### ✨ Suggesting Features

We love feature suggestions! Please:

1. **Check existing feature requests** first
2. **Describe the use case** clearly
3. **Explain why this feature would be beneficial**
4. **Consider implementation complexity**

### 🔧 Code Contributions

#### Prerequisites

- Rust 1.70+
- Git with submodules support
- CMake (for whisper.cpp)
- C++ compiler (GCC/Clang/MSVC)

#### Development Setup

1. **Fork and clone the repository**:
   ```bash
   git clone --recursive https://github.com/YOUR_USERNAME/whisper-rust.git
   cd whisper-rust
   ```

2. **Create a development branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Install development dependencies**:
   ```bash
   # Install clippy and rustfmt
   rustup component add clippy rustfmt
   ```

4. **Build and test**:
   ```bash
   cargo build
   cargo test
   cargo clippy
   cargo fmt --check
   ```

#### Code Style Guidelines

##### Rust Code Standards

- **Follow Rust conventions**: Use `cargo fmt` for formatting
- **Lint with Clippy**: Ensure `cargo clippy` passes without warnings
- **Error handling**: Use `Result<T, E>` pattern, avoid `unwrap()` in library code
- **Documentation**: All public functions must have doc comments
- **Testing**: Add comprehensive tests for new functionality

##### Multi-language Support

- **English comments**: All code comments should be in English
- **Bilingual documentation**: User-facing documentation should include Chinese translations where appropriate
- **Unicode handling**: Ensure proper Unicode support for Chinese characters
- **Language detection**: Implement automatic language detection for transcribed text

##### Example Function Documentation

```rust
/**
 * Process Chinese audio samples for Whisper model compatibility
 * 
 * Converts integer audio samples to floating-point format and handles
 * stereo-to-mono conversion while preserving Chinese speech characteristics.
 * 
 * @param samples - Raw integer audio samples from WAV file
 * @return Result containing processed f32 mono samples or error message
 * 
 * # Examples
 * 
 * ```rust
 * let samples = vec![1000i16, -1000i16, 500i16, -500i16];
 * let processed = process_chinese_audio_samples(&samples)?;
 * assert_eq!(processed.len(), 2); // Stereo converted to mono
 * ```
 * 
 * # Errors
 * 
 * Returns error if:
 * - Audio conversion fails
 * - Sample format is invalid
 * - Memory allocation fails
 */
fn process_chinese_audio_samples(samples: &[i16]) -> Result<Vec<f32>, String> {
    // Implementation
}
```

#### Testing Guidelines

##### Test Coverage Requirements

- **Unit tests**: All public functions must have unit tests
- **Integration tests**: Complex workflows should have integration tests
- **Error path testing**: Test error conditions and edge cases
- **Multi-language testing**: Include tests for Chinese text processing

##### Test Structure

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chinese_character_detection() {
        // Test Chinese character detection
        assert!(contains_chinese("你好世界"));
        assert!(!contains_chinese("Hello World"));
        assert!(contains_chinese("Hello 世界")); // Mixed content
    }

    #[test]
    fn test_audio_processing_edge_cases() {
        // Test with empty audio
        let empty_samples = vec![];
        assert!(process_audio_samples(&empty_samples).is_err());
        
        // Test with maximum values
        let max_samples = vec![i16::MAX, i16::MIN];
        assert!(process_audio_samples(&max_samples).is_ok());
    }

    #[test]
    fn test_error_handling() {
        // Ensure proper error propagation
        let invalid_path = "nonexistent/model.bin";
        let result = load_whisper_model(invalid_path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Failed to load model"));
    }
}
```

#### Logging and Error Handling

##### Logging Standards

```rust
/**
 * Log with timestamp for debugging and monitoring
 * 
 * @param level - Log level (INFO, WARN, ERROR, DEBUG)
 * @param message - Message to log (should be descriptive)
 */
fn log_with_timestamp(level: &str, message: &str) {
    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    
    println!("[{}] {}: {}", timestamp, level, message);
}
```

##### Error Handling Patterns

```rust
// Good: Descriptive error messages
fn load_audio_file(path: &str) -> Result<Vec<f32>, String> {
    if !Path::new(path).exists() {
        let error_msg = format!("Audio file does not exist: {}", path);
        log_with_timestamp("ERROR", &error_msg);
        return Err(error_msg);
    }
    
    // ... implementation
}

// Bad: Generic error messages
fn load_audio_file(path: &str) -> Result<Vec<f32>, String> {
    let file = File::open(path).map_err(|_| "Error")?; // Too generic!
    // ... 
}
```

#### Performance Guidelines

- **Memory efficiency**: Avoid unnecessary allocations in hot paths
- **Error path optimization**: Error handling should be fast
- **Large file handling**: Support streaming for large audio files
- **Unicode performance**: Efficient Chinese character processing

#### Documentation Requirements

##### Code Documentation

- **File headers**: All source files must have comprehensive headers
- **Function documentation**: Use JSDoc-style comments for all public functions
- **Example code**: Include working examples in documentation
- **Error documentation**: Document all possible error conditions

##### README Updates

When adding features, update:

- Feature list in README.md
- Usage examples
- Performance benchmarks (if applicable)
- Supported language list

#### Pull Request Process

##### Before Submitting

1. **Run all checks**:
   ```bash
   cargo test
   cargo clippy -- -D warnings
   cargo fmt --check
   ```

2. **Update documentation** as needed
3. **Add tests** for new functionality
4. **Update CHANGELOG.md** with your changes

##### PR Description Template

```markdown
## 🔧 Changes Made

Brief description of what this PR does.

## 🧪 Testing

- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing performed

## 📝 Documentation

- [ ] Code comments updated
- [ ] README.md updated (if needed)
- [ ] Examples updated (if needed)

## 🌍 Multi-language Support

- [ ] Chinese character handling tested
- [ ] Unicode processing verified
- [ ] Language detection working

## ✅ Checklist

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] No new clippy warnings
- [ ] Documentation is clear and complete
```

##### Review Process

1. **Automated checks** must pass (CI/CD)
2. **Code review** by maintainers
3. **Testing** on multiple platforms
4. **Documentation review** for clarity

#### Release Process

##### Version Numbering

- **Major**: Breaking changes (v1.0.0 → v2.0.0)
- **Minor**: New features, backward compatible (v1.0.0 → v1.1.0)
- **Patch**: Bug fixes, backward compatible (v1.0.0 → v1.0.1)

##### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in Cargo.toml
- [ ] Git tag created
- [ ] GitHub release published

## 🎯 Priority Areas for Contribution

### High Priority

- **Chinese language optimization**: Improve Chinese speech recognition accuracy
- **Performance improvements**: Optimize audio processing pipelines
- **Error handling**: Enhance error messages and recovery
- **Documentation**: Improve bilingual documentation

### Medium Priority

- **Additional languages**: Support for more languages
- **GPU acceleration**: CUDA/Metal/Vulkan optimizations
- **Audio formats**: Support for more input formats
- **Streaming**: Real-time audio processing

### Low Priority

- **UI improvements**: Better CLI interfaces
- **Benchmarking**: Performance comparison tools
- **Examples**: More use case examples

## 🤔 Questions or Need Help?

- **Create an issue**: For bugs, feature requests, or questions
- **Discussion board**: For general discussions about the project
- **Code review**: For specific code questions

## 📜 Code of Conduct

- **Be respectful**: Treat all contributors with respect
- **Be inclusive**: Welcome contributors from all backgrounds
- **Be constructive**: Provide helpful feedback
- **Be patient**: Remember that everyone is learning

## 🙏 Recognition

All contributors will be:

- **Listed in CONTRIBUTORS.md**
- **Mentioned in release notes**
- **Credited in documentation**

---

Thank you for contributing to Enhanced Whisper-RS! Together, we're building the best multi-language speech recognition library for Rust. 🚀 