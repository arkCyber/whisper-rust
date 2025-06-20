use std::ffi::{c_int, NulError};
use std::str::Utf8Error;

/// If you have not configured a logging trampoline with [crate::whisper_sys_log::install_whisper_log_trampoline] or
/// [crate::whisper_sys_tracing::install_whisper_tracing_trampoline],
/// then `whisper.cpp`'s errors will be output to stderr,
/// so you can check there for more information upon receiving a `WhisperError`.
#[derive(Debug, Copy, Clone)]
pub enum WhisperError {
    /// Failed to create a new context.
    InitError,
    /// User didn't initialize spectrogram
    SpectrogramNotInitialized,
    /// Encode was not called.
    EncodeNotComplete,
    /// Decode was not called.
    DecodeNotComplete,
    /// Failed to calculate the spectrogram for some reason.
    UnableToCalculateSpectrogram,
    /// Failed to evaluate model.
    UnableToCalculateEvaluation,
    /// Failed to run the encoder
    FailedToEncode,
    /// Failed to run the decoder
    FailedToDecode,
    /// Invalid number of mel bands.
    InvalidMelBands,
    /// Invalid thread count
    InvalidThreadCount,
    /// Invalid UTF-8 detected in a string from Whisper.
    InvalidUtf8 {
        error_len: Option<usize>,
        valid_up_to: usize,
    },
    /// A null byte was detected in a user-provided string.
    NullByteInString { idx: usize },
    /// Whisper returned a null pointer.
    NullPointer,
    /// Generic whisper error. Varies depending on the function.
    GenericError(c_int),
    /// Whisper failed to convert the provided text into tokens.
    InvalidText,
    /// Creating a state pointer failed. Check stderr for more information.
    FailedToCreateState,
    /// No samples were provided.
    NoSamples,
    /// Input and output slices were not the same length.
    InputOutputLengthMismatch { input_len: usize, output_len: usize },
    /// Input slice was not an even number of samples.
    HalfSampleMissing(usize),
}

impl WhisperError {
    /// Log error with context information including timestamp
    pub fn log_error(&self, context: &str) {
        use crate::common_logging::generic_error;
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        generic_error!("[{}] WhisperError in {}: {}", timestamp, context, self);
    }
    
    /// Get error severity level for logging purposes
    pub fn severity(&self) -> &'static str {
        match self {
            WhisperError::InitError | 
            WhisperError::FailedToCreateState |
            WhisperError::NullPointer => "CRITICAL",
            
            WhisperError::SpectrogramNotInitialized |
            WhisperError::EncodeNotComplete |
            WhisperError::DecodeNotComplete |
            WhisperError::UnableToCalculateSpectrogram |
            WhisperError::UnableToCalculateEvaluation |
            WhisperError::FailedToEncode |
            WhisperError::FailedToDecode => "ERROR",
            
            WhisperError::InvalidMelBands |
            WhisperError::InvalidThreadCount |
            WhisperError::InvalidText |
            WhisperError::InputOutputLengthMismatch { .. } |
            WhisperError::HalfSampleMissing(_) => "WARNING",
            
            WhisperError::InvalidUtf8 { .. } |
            WhisperError::NullByteInString { .. } |
            WhisperError::NoSamples |
            WhisperError::GenericError(_) => "INFO",
        }
    }

    /// Get error code for interoperability
    pub fn error_code(&self) -> i32 {
        match self {
            WhisperError::InitError => 1001,
            WhisperError::SpectrogramNotInitialized => 1002,
            WhisperError::EncodeNotComplete => 1003,
            WhisperError::DecodeNotComplete => 1004,
            WhisperError::UnableToCalculateSpectrogram => 1005,
            WhisperError::UnableToCalculateEvaluation => 1006,
            WhisperError::FailedToEncode => 1007,
            WhisperError::FailedToDecode => 1008,
            WhisperError::InvalidMelBands => 1009,
            WhisperError::InvalidThreadCount => 1010,
            WhisperError::InvalidUtf8 { .. } => 1011,
            WhisperError::NullByteInString { .. } => 1012,
            WhisperError::NullPointer => 1013,
            WhisperError::GenericError(code) => *code,
            WhisperError::InvalidText => 1015,
            WhisperError::FailedToCreateState => 1016,
            WhisperError::NoSamples => 1017,
            WhisperError::InputOutputLengthMismatch { .. } => 1018,
            WhisperError::HalfSampleMissing(_) => 1019,
        }
    }
}

impl From<Utf8Error> for WhisperError {
    fn from(e: Utf8Error) -> Self {
        let error = Self::InvalidUtf8 {
            error_len: e.error_len(),
            valid_up_to: e.valid_up_to(),
        };
        error.log_error("UTF-8 conversion");
        error
    }
}

impl From<NulError> for WhisperError {
    fn from(e: NulError) -> Self {
        let error = Self::NullByteInString {
            idx: e.nul_position(),
        };
        error.log_error("null byte conversion");
        error
    }
}

impl std::fmt::Display for WhisperError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        use WhisperError::*;
        match self {
            InitError => write!(f, "Failed to create a new whisper context."),
            SpectrogramNotInitialized => write!(f, "User didn't initialize spectrogram."),
            EncodeNotComplete => write!(f, "Encode was not called."),
            DecodeNotComplete => write!(f, "Decode was not called."),
            UnableToCalculateSpectrogram => {
                write!(f, "Failed to calculate the spectrogram for some reason.")
            }
            UnableToCalculateEvaluation => write!(f, "Failed to evaluate model."),
            FailedToEncode => write!(f, "Failed to run the encoder."),
            FailedToDecode => write!(f, "Failed to run the decoder."),
            InvalidMelBands => write!(f, "Invalid number of mel bands."),
            InvalidThreadCount => write!(f, "Invalid thread count."),
            InvalidUtf8 {
                valid_up_to,
                error_len: Some(len),
            } => write!(
                f,
                "Invalid UTF-8 detected in a string from Whisper. Index: {}, Length: {}.",
                valid_up_to, len
            ),
            InvalidUtf8 {
                valid_up_to,
                error_len: None,
            } => write!(
                f,
                "Invalid UTF-8 detected in a string from Whisper. Index: {}.",
                valid_up_to
            ),
            NullByteInString { idx } => write!(
                f,
                "A null byte was detected in a user-provided string. Index: {}",
                idx
            ),
            NullPointer => write!(f, "Whisper returned a null pointer."),
            InvalidText => write!(
                f,
                "Whisper failed to convert the provided text into tokens."
            ),
            FailedToCreateState => write!(f, "Creating a state pointer failed."),
            GenericError(c_int) => write!(
                f,
                "Generic whisper error. Varies depending on the function. Error code: {}",
                c_int
            ),
            NoSamples => write!(f, "Input sample buffer was empty."),
            InputOutputLengthMismatch {
                output_len,
                input_len,
            } => {
                write!(
                    f,
                    "Input and output slices were not the same length. Input: {}, Output: {}",
                    input_len, output_len
                )
            }
            HalfSampleMissing(size) => {
                write!(
                    f,
                    "Input slice was not an even number of samples, got {}, expected {}",
                    size,
                    size + 1
                )
            }
        }
    }
}

impl std::error::Error for WhisperError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    
    #[test]
    fn test_error_display_formatting() {
        let init_error = WhisperError::InitError;
        assert_eq!(init_error.to_string(), "Failed to create a new whisper context.");
        
        let utf8_error = WhisperError::InvalidUtf8 {
            valid_up_to: 5,
            error_len: Some(2),
        };
        assert_eq!(
            utf8_error.to_string(),
            "Invalid UTF-8 detected in a string from Whisper. Index: 5, Length: 2."
        );
        
        let utf8_error_no_len = WhisperError::InvalidUtf8 {
            valid_up_to: 3,
            error_len: None,
        };
        assert_eq!(
            utf8_error_no_len.to_string(),
            "Invalid UTF-8 detected in a string from Whisper. Index: 3."
        );
        
        let length_mismatch = WhisperError::InputOutputLengthMismatch {
            input_len: 100,
            output_len: 50,
        };
        assert_eq!(
            length_mismatch.to_string(),
            "Input and output slices were not the same length. Input: 100, Output: 50"
        );
    }
    
    #[test]
    fn test_error_codes() {
        assert_eq!(WhisperError::InitError.error_code(), 1001);
        assert_eq!(WhisperError::SpectrogramNotInitialized.error_code(), 1002);
        assert_eq!(WhisperError::GenericError(42).error_code(), 42);
        assert_eq!(WhisperError::HalfSampleMissing(10).error_code(), 1019);
    }
    
    #[test]
    fn test_error_severity() {
        assert_eq!(WhisperError::InitError.severity(), "CRITICAL");
        assert_eq!(WhisperError::NullPointer.severity(), "CRITICAL");
        assert_eq!(WhisperError::FailedToEncode.severity(), "ERROR");
        assert_eq!(WhisperError::InvalidMelBands.severity(), "WARNING");
        assert_eq!(WhisperError::NoSamples.severity(), "INFO");
    }
    
    #[test]
    fn test_null_error_conversion() {
        let test_string = "test\0embedded";
        let result = CString::new(test_string);
        assert!(result.is_err());
        
        if let Err(nul_error) = result {
            let whisper_error: WhisperError = nul_error.into();
            match whisper_error {
                WhisperError::NullByteInString { idx } => {
                    assert_eq!(idx, 4); // Position of null byte
                }
                _ => panic!("Expected NullByteInString error"),
            }
        }
    }
    
    #[test]
    fn test_utf8_error_conversion() {
        // Create invalid UTF-8 bytes
        let invalid_utf8 = vec![0xFF, 0xFE, 0xFD];
        let result = std::str::from_utf8(&invalid_utf8);
        assert!(result.is_err());
        
        if let Err(utf8_error) = result {
            let whisper_error: WhisperError = utf8_error.into();
            match whisper_error {
                WhisperError::InvalidUtf8 { valid_up_to, error_len } => {
                    assert_eq!(valid_up_to, 0);
                    assert!(error_len.is_some());
                }
                _ => panic!("Expected InvalidUtf8 error"),
            }
        }
    }
    
    #[test]
    fn test_error_debug_format() {
        let error = WhisperError::GenericError(123);
        let debug_str = format!("{:?}", error);
        assert!(debug_str.contains("GenericError"));
        assert!(debug_str.contains("123"));
    }
    
    #[test]
    fn test_error_clone_and_copy() {
        let original = WhisperError::InvalidThreadCount;
        let cloned = original.clone();
        let copied = original;
        
        // All should be equal
        assert_eq!(format!("{:?}", original), format!("{:?}", cloned));
        assert_eq!(format!("{:?}", original), format!("{:?}", copied));
    }
    
    #[test]
    fn test_all_error_variants_display() {
        let errors = vec![
            WhisperError::InitError,
            WhisperError::SpectrogramNotInitialized,
            WhisperError::EncodeNotComplete,
            WhisperError::DecodeNotComplete,
            WhisperError::UnableToCalculateSpectrogram,
            WhisperError::UnableToCalculateEvaluation,
            WhisperError::FailedToEncode,
            WhisperError::FailedToDecode,
            WhisperError::InvalidMelBands,
            WhisperError::InvalidThreadCount,
            WhisperError::NullPointer,
            WhisperError::InvalidText,
            WhisperError::FailedToCreateState,
            WhisperError::NoSamples,
            WhisperError::GenericError(-1),
            WhisperError::HalfSampleMissing(7),
        ];
        
        for error in errors {
            let display = error.to_string();
            assert!(!display.is_empty(), "Error display should not be empty for {:?}", error);
            assert!(error.error_code() != 0, "Error code should not be zero for {:?}", error);
            assert!(!error.severity().is_empty(), "Severity should not be empty for {:?}", error);
        }
    }
    
    #[test]
    fn test_log_error_does_not_panic() {
        let error = WhisperError::InitError;
        // This should not panic even if logging is not configured
        error.log_error("test context");
    }
}
