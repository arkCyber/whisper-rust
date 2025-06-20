use std::{ffi::CStr, os::raw::c_int};
use whisper_rs_sys::{
    ggml_backend_buffer_type_t, ggml_backend_vk_buffer_type, ggml_backend_vk_get_device_count,
    ggml_backend_vk_get_device_description, ggml_backend_vk_get_device_memory,
};

/// VRAM information for a Vulkan device
/// 
/// # Fields
/// * `free` - Amount of free VRAM in bytes
/// * `total` - Total amount of VRAM in bytes
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VKVram {
    pub free: usize,
    pub total: usize,
}

impl VKVram {
    /// Create a new VKVram instance with validation
    /// 
    /// # Arguments
    /// * `free` - Free VRAM in bytes
    /// * `total` - Total VRAM in bytes
    /// 
    /// # Returns
    /// `Some(VKVram)` if valid, `None` if invalid (free > total)
    pub fn new(free: usize, total: usize) -> Option<Self> {
        if free <= total {
            Some(Self { free, total })
        } else {
            use crate::common_logging::generic_warn;
            generic_warn!("Invalid VRAM values: free {} > total {}", free, total);
            None
        }
    }
    
    /// Get the used VRAM amount
    /// 
    /// # Returns
    /// Amount of used VRAM in bytes
    pub fn used(&self) -> usize {
        self.total.saturating_sub(self.free)
    }
    
    /// Get the percentage of used VRAM
    /// 
    /// # Returns
    /// Percentage of used VRAM (0.0 to 100.0)
    pub fn usage_percentage(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.used() as f64 / self.total as f64) * 100.0
        }
    }
    
    /// Check if VRAM usage is above a threshold
    /// 
    /// # Arguments
    /// * `threshold` - Threshold percentage (0.0 to 100.0)
    /// 
    /// # Returns
    /// `true` if usage is above threshold
    pub fn is_usage_above(&self, threshold: f64) -> bool {
        self.usage_percentage() > threshold
    }
    
    /// Format VRAM as human-readable string
    /// 
    /// # Returns
    /// String representation like "1.5GB / 8GB (18.75%)"
    pub fn to_human_readable(&self) -> String {
        let free_gb = self.free as f64 / (1024.0 * 1024.0 * 1024.0);
        let total_gb = self.total as f64 / (1024.0 * 1024.0 * 1024.0);
        let usage = self.usage_percentage();
        
        format!("{:.1}GB / {:.1}GB ({:.1}%)", 
                total_gb - free_gb, total_gb, usage)
    }
}

/// Human-readable device information
/// 
/// # Fields
/// * `id` - Device ID
/// * `name` - Device name
/// * `vram` - VRAM information
/// * `buf_type` - Buffer type for creating buffers
#[derive(Debug, Clone)]
pub struct VkDeviceInfo {
    pub id: i32,
    pub name: String,
    pub vram: VKVram,
    /// Buffer type to pass to `whisper::Backend::create_buffer`
    pub buf_type: ggml_backend_buffer_type_t,
}

impl VkDeviceInfo {
    /// Check if this device is suitable for the given memory requirement
    /// 
    /// # Arguments
    /// * `required_memory` - Required memory in bytes
    /// 
    /// # Returns
    /// `true` if device has enough free memory
    pub fn can_allocate(&self, required_memory: usize) -> bool {
        self.vram.free >= required_memory
    }
    
    /// Get a detailed description of the device
    /// 
    /// # Returns
    /// Formatted string with device details
    pub fn detailed_description(&self) -> String {
        format!("Vulkan Device {} ({}): {}", 
                self.id, self.name, self.vram.to_human_readable())
    }
    
    /// Check if this is likely a discrete GPU based on VRAM
    /// 
    /// # Returns
    /// `true` if likely discrete (>= 1GB VRAM), `false` if integrated
    pub fn is_likely_discrete(&self) -> bool {
        self.vram.total >= 1024 * 1024 * 1024 // 1GB
    }
}

/// Result type for Vulkan operations
pub type VulkanResult<T> = Result<T, VulkanError>;

/// Errors that can occur with Vulkan operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VulkanError {
    /// No Vulkan devices found
    NoDevicesFound,
    /// Invalid device ID
    InvalidDeviceId(i32),
    /// Device name extraction failed
    NameExtractionFailed(i32),
    /// Memory query failed
    MemoryQueryFailed(i32),
    /// Buffer type query failed
    BufferTypeQueryFailed(i32),
}

impl std::fmt::Display for VulkanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VulkanError::NoDevicesFound => write!(f, "No Vulkan devices found"),
            VulkanError::InvalidDeviceId(id) => write!(f, "Invalid Vulkan device ID: {}", id),
            VulkanError::NameExtractionFailed(id) => write!(f, "Failed to extract name for device {}", id),
            VulkanError::MemoryQueryFailed(id) => write!(f, "Failed to query memory for device {}", id),
            VulkanError::BufferTypeQueryFailed(id) => write!(f, "Failed to get buffer type for device {}", id),
        }
    }
}

impl std::error::Error for VulkanError {}

/// Enumerate every physical GPU ggml can see.
///
/// Note: integrated GPUs are returned *after* discrete ones,
/// mirroring ggml's C logic.
/// 
/// # Returns
/// Vector of device information, or empty vector if no devices found
/// 
/// # Examples
/// ```no_run
/// # use whisper_rs::vulkan::list_devices;
/// let devices = list_devices();
/// for device in devices {
///     println!("Found Vulkan device: {}", device.detailed_description());
/// }
/// ```
pub fn list_devices() -> Vec<VkDeviceInfo> {
    list_devices_with_logging().unwrap_or_else(|e| {
        use crate::common_logging::generic_warn;
        generic_warn!("Failed to enumerate Vulkan devices: {}", e);
        Vec::new()
    })
}

/// Enumerate Vulkan devices with detailed error reporting
/// 
/// # Returns
/// `Ok(Vec<VkDeviceInfo>)` on success, `Err(VulkanError)` on failure
pub fn list_devices_with_logging() -> VulkanResult<Vec<VkDeviceInfo>> {
    use crate::common_logging::{generic_info, generic_debug, generic_error};
    
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    
    generic_debug!("[{}] Starting Vulkan device enumeration", timestamp);
    
    unsafe {
        let n = ggml_backend_vk_get_device_count();
        generic_info!("[{}] Found {} Vulkan devices", timestamp, n);
        
        if n == 0 {
            return Err(VulkanError::NoDevicesFound);
        }
        
        let mut devices = Vec::with_capacity(n as usize);
        
        for id in 0..n {
            generic_debug!("[{}] Querying device {}", timestamp, id);
            
            // Get device name
            let mut tmp = [0i8; 256];
            ggml_backend_vk_get_device_description(id as c_int, tmp.as_mut_ptr(), tmp.len());
            
            let name = CStr::from_ptr(tmp.as_ptr())
                .to_string_lossy()
                .into_owned();
                
            if name.trim().is_empty() {
                generic_error!("[{}] Failed to get name for device {}", timestamp, id);
                return Err(VulkanError::NameExtractionFailed(id));
            }
            
            // Get memory information
            let mut free = 0usize;
            let mut total = 0usize;
            ggml_backend_vk_get_device_memory(id, &mut free, &mut total);
            
            let vram = VKVram::new(free, total)
                .ok_or(VulkanError::MemoryQueryFailed(id))?;
            
            // Get buffer type
            let buf_type = ggml_backend_vk_buffer_type(id as usize);
            if buf_type.is_null() {
                generic_error!("[{}] Failed to get buffer type for device {}", timestamp, id);
                return Err(VulkanError::BufferTypeQueryFailed(id));
            }
            
            let device_info = VkDeviceInfo {
                id,
                name: name.clone(),
                vram,
                buf_type,
            };
            
            generic_info!("[{}] Device {}: {}", timestamp, id, device_info.detailed_description());
            devices.push(device_info);
        }
        
        generic_info!("[{}] Successfully enumerated {} devices", timestamp, devices.len());
        Ok(devices)
    }
}

/// Get device by ID with error handling
/// 
/// # Arguments
/// * `device_id` - The device ID to query
/// 
/// # Returns
/// `Ok(VkDeviceInfo)` if found, `Err(VulkanError)` if not found
pub fn get_device_by_id(device_id: i32) -> VulkanResult<VkDeviceInfo> {
    let devices = list_devices_with_logging()?;
    devices.into_iter()
        .find(|d| d.id == device_id)
        .ok_or(VulkanError::InvalidDeviceId(device_id))
}

/// Find the best device for a given memory requirement
/// 
/// # Arguments
/// * `required_memory` - Required memory in bytes
/// 
/// # Returns
/// `Some(VkDeviceInfo)` for the best suitable device, `None` if none found
pub fn find_best_device(required_memory: usize) -> Option<VkDeviceInfo> {
    let devices = list_devices();
    
    // Filter devices that can handle the requirement
    let mut suitable: Vec<_> = devices.into_iter()
        .filter(|d| d.can_allocate(required_memory))
        .collect();
    
    if suitable.is_empty() {
        return None;
    }
    
    // Sort by preference: discrete first, then by free memory
    suitable.sort_by(|a, b| {
        match (a.is_likely_discrete(), b.is_likely_discrete()) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => b.vram.free.cmp(&a.vram.free), // More free memory is better
        }
    });
    
    suitable.into_iter().next()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vkvram_creation() {
        // Valid VRAM
        let vram = VKVram::new(1024, 2048).unwrap();
        assert_eq!(vram.free, 1024);
        assert_eq!(vram.total, 2048);
        
        // Invalid VRAM (free > total)
        assert!(VKVram::new(2048, 1024).is_none());
        
        // Edge case: free == total
        let vram = VKVram::new(1024, 1024).unwrap();
        assert_eq!(vram.free, 1024);
        assert_eq!(vram.total, 1024);
    }
    
    #[test]
    fn test_vkvram_calculations() {
        let vram = VKVram::new(1024, 4096).unwrap();
        
        assert_eq!(vram.used(), 3072);
        assert!((vram.usage_percentage() - 75.0).abs() < 0.001);
        assert!(vram.is_usage_above(50.0));
        assert!(!vram.is_usage_above(80.0));
    }
    
    #[test]
    fn test_vkvram_zero_total() {
        let vram = VKVram::new(0, 0).unwrap();
        assert_eq!(vram.used(), 0);
        assert_eq!(vram.usage_percentage(), 0.0);
        assert!(!vram.is_usage_above(50.0));
    }
    
    #[test]
    fn test_vkvram_human_readable() {
        let vram = VKVram::new(1024 * 1024 * 1024, 2 * 1024 * 1024 * 1024); // 1GB free, 2GB total
        let readable = vram.to_human_readable();
        assert!(readable.contains("1.0GB"));
        assert!(readable.contains("2.0GB"));
        assert!(readable.contains("50.0%"));
    }
    
    #[test]
    fn test_vkvram_equality() {
        let vram1 = VKVram::new(1024, 2048).unwrap();
        let vram2 = VKVram::new(1024, 2048).unwrap();
        let vram3 = VKVram::new(512, 2048).unwrap();
        
        assert_eq!(vram1, vram2);
        assert_ne!(vram1, vram3);
    }
    
    #[test]
    fn test_device_info_capabilities() {
        let vram = VKVram::new(1024 * 1024 * 1024, 4 * 1024 * 1024 * 1024); // 1GB free, 4GB total
        let device = VkDeviceInfo {
            id: 0,
            name: "Test GPU".to_string(),
            vram,
            buf_type: std::ptr::null_mut(),
        };
        
        assert!(device.can_allocate(512 * 1024 * 1024)); // 512MB
        assert!(!device.can_allocate(2 * 1024 * 1024 * 1024)); // 2GB
        assert!(device.is_likely_discrete());
        
        let description = device.detailed_description();
        assert!(description.contains("Test GPU"));
        assert!(description.contains("0"));
    }
    
    #[test]
    fn test_integrated_gpu_detection() {
        let vram = VKVram::new(256 * 1024 * 1024, 512 * 1024 * 1024); // 256MB free, 512MB total
        let device = VkDeviceInfo {
            id: 1,
            name: "Integrated GPU".to_string(),
            vram,
            buf_type: std::ptr::null_mut(),
        };
        
        assert!(!device.is_likely_discrete());
    }
    
    #[test]
    fn test_vulkan_error_display() {
        let errors = vec![
            VulkanError::NoDevicesFound,
            VulkanError::InvalidDeviceId(42),
            VulkanError::NameExtractionFailed(1),
            VulkanError::MemoryQueryFailed(2),
            VulkanError::BufferTypeQueryFailed(3),
        ];
        
        for error in errors {
            let display = error.to_string();
            assert!(!display.is_empty());
            
            // Test Debug format too
            let debug = format!("{:?}", error);
            assert!(!debug.is_empty());
        }
    }
    
    #[test]
    fn test_vulkan_error_equality() {
        assert_eq!(VulkanError::NoDevicesFound, VulkanError::NoDevicesFound);
        assert_eq!(VulkanError::InvalidDeviceId(42), VulkanError::InvalidDeviceId(42));
        assert_ne!(VulkanError::InvalidDeviceId(42), VulkanError::InvalidDeviceId(43));
    }
    
    #[test]
    fn enumerate_must_not_panic() {
        let _ = list_devices();
    }

    #[test]
    fn sane_device_info() {
        let gpus = list_devices();
        let mut seen = std::collections::HashSet::new();

        for dev in &gpus {
            assert!(seen.insert(dev.id), "duplicated id {}", dev.id);
            assert!(!dev.name.trim().is_empty(), "GPU {} has empty name", dev.id);
            assert!(
                dev.vram.total >= dev.vram.free,
                "GPU {} total < free",
                dev.id
            );
        }
    }
    
    #[test]
    fn test_device_enumeration_with_logging() {
        // This should not panic regardless of system state
        match list_devices_with_logging() {
            Ok(devices) => {
                println!("Found {} Vulkan devices", devices.len());
                for device in devices {
                    assert!(!device.name.is_empty());
                    assert!(device.vram.total >= device.vram.free);
                }
            }
            Err(VulkanError::NoDevicesFound) => {
                println!("No Vulkan devices found (expected on some systems)");
            }
            Err(e) => {
                println!("Vulkan enumeration error: {}", e);
            }
        }
    }
    
    #[test]
    fn test_device_by_id() {
        let devices = list_devices();
        if let Some(first_device) = devices.first() {
            // Test valid ID
            let found = get_device_by_id(first_device.id);
            match found {
                Ok(device) => assert_eq!(device.id, first_device.id),
                Err(VulkanError::NoDevicesFound) => {
                    // Acceptable if no devices
                }
                Err(e) => panic!("Unexpected error: {}", e),
            }
        }
        
        // Test invalid ID
        let invalid_result = get_device_by_id(-999);
        assert!(invalid_result.is_err());
    }
    
    #[test]
    fn test_find_best_device() {
        let devices = list_devices();
        if devices.is_empty() {
            // No devices to test
            assert!(find_best_device(1024).is_none());
            return;
        }
        
        // Test with very small requirement - should find a device
        let best = find_best_device(1024); // 1KB
        if let Some(device) = best {
            assert!(device.can_allocate(1024));
        }
        
        // Test with impossibly large requirement
        let best_large = find_best_device(usize::MAX);
        assert!(best_large.is_none());
    }
    
    #[test]
    fn test_boundary_conditions() {
        // Test zero memory requirement
        let best = find_best_device(0);
        if let Some(device) = best {
            assert!(device.can_allocate(0));
        }
        
        // Test VRAM edge cases
        let vram_zero = VKVram::new(0, 0).unwrap();
        assert_eq!(vram_zero.used(), 0);
        
        let vram_max = VKVram::new(0, usize::MAX).unwrap();
        assert_eq!(vram_max.used(), usize::MAX);
    }
}
