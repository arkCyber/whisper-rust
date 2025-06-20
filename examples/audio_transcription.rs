/**
 * Enhanced Audio Transcription Example for Whisper-RS
 * 
 * This example demonstrates advanced features of whisper-rs including:
 * - DTW (Dynamic Time Warping) token level timestamps
 * - Audio file processing with format validation
 * - Token-level analysis and DTW timestamp extraction
 * - File output generation for transcription results
 * - Comprehensive error handling and logging
 * 
 * Features:
 * - DTW token level timestamp configuration (model preset and custom)
 * - Audio format validation and conversion
 * - Detailed logging with timestamps for all operations
 * - Token-level data extraction and analysis
 * - Transcript file generation
 * - Enhanced error handling throughout the process
 * 
 * Note: This example requires model files and audio files to be present.
 * You need to copy this code into your project and add the dependencies 
 * whisper_rs and hound in your Cargo.toml
 * 
 * Dependencies:
 *   - whisper-rs: Main library for Whisper functionality
 *   - hound: WAV file reading and processing
 * 
 * Author: Enhanced for whisper-rs project
 * Date: 2024
 * License: Same as whisper-rs project
 */

use hound;
use std::fs::File;
use std::io::Write;
use std::time::SystemTime;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/**
 * Print a timestamped log message to stdout
 * 
 * @param level - Log level (INFO, WARN, ERROR, etc.)
 * @param message - The message to log
 */
fn log_with_timestamp(level: &str, message: &str) {
    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    
    println!("[{}] {}: {}", timestamp, level, message);
}

/**
 * Configure DTW (Dynamic Time Warping) parameters for enhanced timestamp accuracy
 * 
 * @return Configured WhisperContextParameters with DTW settings
 */
fn configure_dtw_parameters() -> WhisperContextParameters<'static> {
    let _timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    
    log_with_timestamp("INFO", "Configuring DTW parameters for enhanced timestamps");
    
    let mut context_param = WhisperContextParameters::default();

    // Enable DTW token level timestamp for known model by using model preset
    context_param.dtw_parameters.mode = whisper_rs::DtwMode::ModelPreset {
        model_preset: whisper_rs::DtwModelPreset::BaseEn,
    };
    
    log_with_timestamp("INFO", "DTW mode set to ModelPreset::BaseEn");

    // Example of custom DTW configuration (commented out, but shown for reference)
    // Enable DTW token level timestamp for unknown model by providing custom aheads
    // see details https://github.com/ggerganov/whisper.cpp/pull/1485#discussion_r1519681143
    // values corresponds to ggml-base.en.bin, result will be the same as with DtwModelPreset::BaseEn
    let _custom_aheads = [
        (3, 1),
        (4, 2),
        (4, 3),
        (4, 7),
        (5, 1),
        (5, 2),
        (5, 4),
        (5, 6),
    ]
    .map(|(n_text_layer, n_head)| whisper_rs::DtwAhead {
        n_text_layer,
        n_head,
    });
    
    // Uncomment the following line to use custom DTW configuration instead
    // context_param.dtw_parameters.mode = whisper_rs::DtwMode::Custom {
    //     aheads: &custom_aheads,
    // };
    
    log_with_timestamp("INFO", "DTW parameters configured with 8 custom aheads available for reference");
    
    context_param
}

/**
 * Initialize Whisper context with DTW parameters
 * 
 * @param model_path - Path to the Whisper model file
 * @return Result containing WhisperContext or error message
 */
fn initialize_whisper_context_with_dtw(model_path: &str) -> Result<WhisperContext, String> {
    let _timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    
    log_with_timestamp("INFO", &format!("Loading Whisper model with DTW: {}", model_path));
    
    // Validate model file existence
    if !std::path::Path::new(model_path).exists() {
        let error_msg = format!("Model file does not exist: {}", model_path);
        log_with_timestamp("ERROR", &error_msg);
        return Err(error_msg);
    }
    
    let context_param = configure_dtw_parameters();
    
    let ctx = WhisperContext::new_with_params(model_path, context_param)
        .map_err(|e| {
            let error_msg = format!("Failed to load model: {}", e);
            log_with_timestamp("ERROR", &error_msg);
            error_msg
        })?;
    
    log_with_timestamp("INFO", "Whisper model with DTW loaded successfully");
    
    Ok(ctx)
}

/**
 * Configure Whisper parameters for transcription with token timestamps
 * 
 * @return Configured FullParams for Whisper with enhanced settings
 */
fn configure_whisper_params_with_tokens() -> FullParams<'static, 'static> {
    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    
    log_with_timestamp("INFO", "Configuring Whisper parameters with token timestamps");
    
    // Create params object for running the model
    // The number of past samples to consider defaults to 0
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 });

    // Edit params as needed
    // Set the number of threads to use to 1
    params.set_n_threads(1);
    log_with_timestamp("INFO", "Set number of threads to 1");
    
    // Enable translation
    params.set_translate(true);
    log_with_timestamp("INFO", "Translation enabled");
    
    // Set the language to translate to English
    params.set_language(Some("en"));
    log_with_timestamp("INFO", "Language set to English");
    
    // Disable anything that prints to stdout to keep our logs clean
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    log_with_timestamp("INFO", "Console output disabled for clean logging");
    
    // Enable token level timestamps
    params.set_token_timestamps(true);
    log_with_timestamp("INFO", "Token level timestamps enabled");
    
    log_with_timestamp("INFO", "Whisper parameters with tokens configured successfully");
    
    params
}

/**
 * Load and validate audio file with format checking
 * 
 * @param audio_path - Path to the audio file
 * @return Result containing (audio_samples, sample_rate, channels) or error message
 */
fn load_and_validate_audio(audio_path: &str) -> Result<(Vec<f32>, u32, u16), String> {
    let _timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    
    log_with_timestamp("INFO", &format!("Loading audio file: {}", audio_path));
    
    // Validate audio file existence
    if !std::path::Path::new(audio_path).exists() {
        let error_msg = format!("Audio file does not exist: {}", audio_path);
        log_with_timestamp("ERROR", &error_msg);
        return Err(error_msg);
    }
    
    // Open the audio file
    let reader = hound::WavReader::open(audio_path)
        .map_err(|e| {
            let error_msg = format!("Failed to open audio file: {}", e);
            log_with_timestamp("ERROR", &error_msg);
            error_msg
        })?;
    
    let hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample,
        ..
    } = reader.spec();
    
    log_with_timestamp("INFO", &format!("Audio specs - Sample rate: {}Hz, Channels: {}, Bits per sample: {}", 
                                       sample_rate, channels, bits_per_sample));
    
    // Validate sample rate
    if sample_rate != 16000 {
        let error_msg = format!("Sample rate must be 16KHz, got {}Hz", sample_rate);
        log_with_timestamp("ERROR", &error_msg);
        return Err(error_msg);
    }
    
    // Validate channel count
    if channels > 2 {
        let error_msg = format!(">2 channels unsupported, got {} channels", channels);
        log_with_timestamp("ERROR", &error_msg);
        return Err(error_msg);
    }
    
    // Convert the audio to floating point samples
    let samples: Result<Vec<i16>, _> = reader
        .into_samples::<i16>()
        .collect();
    
    let samples = samples.map_err(|e| {
        let error_msg = format!("Failed to read audio samples: {}", e);
        log_with_timestamp("ERROR", &error_msg);
        error_msg
    })?;
    
    if samples.is_empty() {
        let error_msg = "Audio file contains no samples";
        log_with_timestamp("ERROR", error_msg);
        return Err(error_msg.to_string());
    }
    
    log_with_timestamp("INFO", &format!("Successfully loaded {} integer samples", samples.len()));
    
    // Convert to float
    let mut audio = vec![0.0f32; samples.len()];
    whisper_rs::convert_integer_to_float_audio(&samples, &mut audio)
        .map_err(|e| {
            let error_msg = format!("Failed to convert integer to float audio: {}", e);
            log_with_timestamp("ERROR", &error_msg);
            error_msg
        })?;
    
    log_with_timestamp("INFO", "Integer to float conversion completed");
    
    // Convert audio to 16KHz mono f32 samples, as required by the model
    // These utilities are provided for convenience, but can be replaced with custom conversion logic
    // SIMD variants of these functions are also available on nightly Rust (see the docs)
    if channels == 2 {
        audio = whisper_rs::convert_stereo_to_mono_audio(&audio)
            .map_err(|e| {
                let error_msg = format!("Failed to convert stereo to mono: {}", e);
                log_with_timestamp("ERROR", &error_msg);
                error_msg
            })?;
        log_with_timestamp("INFO", "Stereo to mono conversion completed");
    }
    
    log_with_timestamp("INFO", &format!("Final audio sample count: {}", audio.len()));
    
    Ok((audio, sample_rate, channels))
}

/**
 * Extract token-level DTW timestamp information
 * 
 * @param state - Whisper state after processing
 * @param segment_index - Index of the segment to analyze
 * @return DTW timestamp for first token or -1 if unavailable
 */
fn extract_first_token_dtw_timestamp(
    state: &whisper_rs::WhisperState,
    segment_index: i32
) -> i64 {
    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    
    match state.full_n_tokens(segment_index) {
        Ok(token_count) => {
            if token_count > 0 {
                match state.full_get_token_data(segment_index, 0) {
                    Ok(token_data) => {
                        log_with_timestamp("DEBUG", &format!("DTW timestamp for segment {}: {}", segment_index, token_data.t_dtw));
                        token_data.t_dtw
                    }
                    Err(e) => {
                        log_with_timestamp("WARN", &format!("Failed to get token data for segment {}: {}", segment_index, e));
                        -1i64
                    }
                }
            } else {
                log_with_timestamp("WARN", &format!("No tokens found in segment {}", segment_index));
                -1i64
            }
        }
        Err(e) => {
            log_with_timestamp("WARN", &format!("Failed to get token count for segment {}: {}", segment_index, e));
            -1i64
        }
    }
}

/**
 * Process transcription results and save to file
 * 
 * @param state - Whisper state after processing
 * @param output_file_path - Path to save transcript file
 * @return Result indicating success or error message
 */
fn process_and_save_transcription(
    state: &whisper_rs::WhisperState,
    output_file_path: &str
) -> Result<(), String> {
    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    
    log_with_timestamp("INFO", &format!("Processing transcription results and saving to: {}", output_file_path));
    
    // Create a file to write the transcript to
    let mut file = File::create(output_file_path)
        .map_err(|e| {
            let error_msg = format!("Failed to create output file: {}", e);
            log_with_timestamp("ERROR", &error_msg);
            error_msg
        })?;
    
    // Get number of segments
    let num_segments = state.full_n_segments()
        .map_err(|e| {
            let error_msg = format!("Failed to get number of segments: {}", e);
            log_with_timestamp("ERROR", &error_msg);
            error_msg
        })?;
    
    log_with_timestamp("INFO", &format!("Found {} transcription segments", num_segments));
    
    if num_segments == 0 {
        log_with_timestamp("WARN", "No transcription segments found");
        return Ok(());
    }
    
    println!("\n=== TRANSCRIPTION RESULTS WITH DTW TIMESTAMPS ===");
    
    // Iterate through the segments of the transcript
    for i in 0..num_segments {
        // Get the transcribed text and timestamps for the current segment
        let segment = state.full_get_segment_text(i)
            .map_err(|e| {
                let error_msg = format!("Failed to get segment text for segment {}: {}", i, e);
                log_with_timestamp("ERROR", &error_msg);
                error_msg
            })?;
        
        let start_timestamp = state.full_get_segment_t0(i)
            .map_err(|e| {
                let error_msg = format!("Failed to get start timestamp for segment {}: {}", i, e);
                log_with_timestamp("ERROR", &error_msg);
                error_msg
            })?;
        
        let end_timestamp = state.full_get_segment_t1(i)
            .map_err(|e| {
                let error_msg = format!("Failed to get end timestamp for segment {}: {}", i, e);
                log_with_timestamp("ERROR", &error_msg);
                error_msg
            })?;

        // Extract DTW timestamp for first token in segment
        let first_token_dtw_ts = extract_first_token_dtw_timestamp(state, i);
        
        // Print the segment to stdout with DTW information
        println!(
            "[{} - {} (DTW: {})]: {}",
            start_timestamp, end_timestamp, first_token_dtw_ts, segment
        );

        // Format the segment information as a string
        let line = format!("[{} - {}]: {}\n", start_timestamp, end_timestamp, segment);

        // Write the segment information to the file
        file.write_all(line.as_bytes())
            .map_err(|e| {
                let error_msg = format!("Failed to write segment to file: {}", e);
                log_with_timestamp("ERROR", &error_msg);
                error_msg
            })?;
    }
    
    println!("=== END TRANSCRIPTION RESULTS ===\n");
    
    log_with_timestamp("INFO", &format!("Transcription saved to file: {}", output_file_path));
    
    Ok(())
}

/**
 * Main function - Enhanced Audio Transcription with DTW timestamps
 * 
 * Loads a context and model, processes an audio file with DTW token-level timestamps,
 * and saves the resulting transcript to a file with comprehensive error handling.
 */
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start_time = SystemTime::now();
    log_with_timestamp("INFO", "Starting enhanced audio transcription with DTW timestamps");
    
    // Configure file paths (these should be updated to actual file paths)
    let model_path = "example/path/to/model/whisper.cpp/models/ggml-base.en.bin";
    let audio_path = "audio.wav";
    let output_path = "transcript.txt";
    
    log_with_timestamp("INFO", &format!("Model path: {}", model_path));
    log_with_timestamp("INFO", &format!("Audio path: {}", audio_path));
    log_with_timestamp("INFO", &format!("Output path: {}", output_path));
    
    // Process execution with comprehensive error handling
    let result = (|| -> Result<(), String> {
        // Initialize Whisper context with DTW parameters
        let ctx = initialize_whisper_context_with_dtw(model_path)?;
        
        // Create a state
        let mut state = ctx.create_state()
            .map_err(|e| {
                let error_msg = format!("Failed to create Whisper state: {}", e);
                log_with_timestamp("ERROR", &error_msg);
                error_msg
            })?;
        
        log_with_timestamp("INFO", "Whisper state created successfully");
        
        // Configure transcription parameters
        let params = configure_whisper_params_with_tokens();
        
        // Load and validate audio
        let (audio, sample_rate, channels) = load_and_validate_audio(audio_path)?;
        
        log_with_timestamp("INFO", "Starting transcription process");
        
        // Run the model
        state.full(params, &audio[..])
            .map_err(|e| {
                let error_msg = format!("Failed to run Whisper model: {}", e);
                log_with_timestamp("ERROR", &error_msg);
                error_msg
            })?;
        
        log_with_timestamp("INFO", "Transcription completed successfully");
        
        // Process results and save to file
        process_and_save_transcription(&state, output_path)?;
        
        Ok(())
    })();
    
    // Handle final result and cleanup
    match result {
        Ok(()) => {
            let duration = start_time.elapsed().unwrap_or_default();
            log_with_timestamp("INFO", &format!("Enhanced audio transcription completed successfully in {:.2} seconds", 
                                               duration.as_secs_f64()));
            Ok(())
        }
        Err(error) => {
            log_with_timestamp("ERROR", &format!("Audio transcription failed: {}", error));
            Err(error.into())
        }
    }
}
