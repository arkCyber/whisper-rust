use whisper_rs_sys::{
    whisper_gretype_WHISPER_GRETYPE_ALT, whisper_gretype_WHISPER_GRETYPE_CHAR,
    whisper_gretype_WHISPER_GRETYPE_CHAR_ALT, whisper_gretype_WHISPER_GRETYPE_CHAR_NOT,
    whisper_gretype_WHISPER_GRETYPE_CHAR_RNG_UPPER, whisper_gretype_WHISPER_GRETYPE_END,
    whisper_gretype_WHISPER_GRETYPE_RULE_REF,
};

#[cfg_attr(any(not(windows), target_env = "gnu"), repr(u32))] // include windows-gnu
#[cfg_attr(all(windows, not(target_env = "gnu")), repr(i32))] // msvc being *special* again
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum WhisperGrammarElementType {
    /// End of rule definition
    End = whisper_gretype_WHISPER_GRETYPE_END,
    /// Start of alternate definition for a rule
    Alternate = whisper_gretype_WHISPER_GRETYPE_ALT,
    /// Non-terminal element: reference to another rule
    RuleReference = whisper_gretype_WHISPER_GRETYPE_RULE_REF,
    /// Terminal element: character (code point)
    Character = whisper_gretype_WHISPER_GRETYPE_CHAR,
    /// Inverse of a character(s)
    NotCharacter = whisper_gretype_WHISPER_GRETYPE_CHAR_NOT,
    /// Modifies a preceding [Self::Character] to be an inclusive range
    CharacterRangeUpper = whisper_gretype_WHISPER_GRETYPE_CHAR_RNG_UPPER,
    /// Modifies a preceding [Self::Character] to add an alternate character to match
    CharacterAlternate = whisper_gretype_WHISPER_GRETYPE_CHAR_ALT,
}

impl WhisperGrammarElementType {
    /// Get the string representation of the grammar element type
    /// 
    /// # Examples
    /// ```
    /// # use whisper_rs::WhisperGrammarElementType;
    /// assert_eq!(WhisperGrammarElementType::End.to_string(), "End");
    /// assert_eq!(WhisperGrammarElementType::Character.to_string(), "Character");
    /// ```
    pub fn to_string(&self) -> &'static str {
        match self {
            WhisperGrammarElementType::End => "End",
            WhisperGrammarElementType::Alternate => "Alternate",
            WhisperGrammarElementType::RuleReference => "RuleReference", 
            WhisperGrammarElementType::Character => "Character",
            WhisperGrammarElementType::NotCharacter => "NotCharacter",
            WhisperGrammarElementType::CharacterRangeUpper => "CharacterRangeUpper",
            WhisperGrammarElementType::CharacterAlternate => "CharacterAlternate",
        }
    }
    
    /// Check if this element type is a terminal element
    /// 
    /// # Returns
    /// `true` if this is a terminal element (Character, NotCharacter, etc.), `false` otherwise
    pub fn is_terminal(&self) -> bool {
        matches!(self, 
            WhisperGrammarElementType::Character |
            WhisperGrammarElementType::NotCharacter |
            WhisperGrammarElementType::CharacterRangeUpper |
            WhisperGrammarElementType::CharacterAlternate
        )
    }
    
    /// Check if this element type is a control element
    /// 
    /// # Returns
    /// `true` if this is a control element (End, Alternate, RuleReference), `false` otherwise
    pub fn is_control(&self) -> bool {
        matches!(self,
            WhisperGrammarElementType::End |
            WhisperGrammarElementType::Alternate |
            WhisperGrammarElementType::RuleReference
        )
    }
    
    /// Validate if a raw value corresponds to a valid grammar element type
    /// 
    /// # Arguments
    /// * `value` - Raw value to validate
    /// 
    /// # Returns
    /// `true` if valid, `false` otherwise
    pub fn is_valid_raw(value: u32) -> bool {
        (0..=6).contains(&value)
    }
}

impl From<whisper_rs_sys::whisper_gretype> for WhisperGrammarElementType {
    fn from(value: whisper_rs_sys::whisper_gretype) -> Self {
        assert!(
            Self::is_valid_raw(value),
            "Invalid WhisperGrammarElementType value: {}",
            value
        );

        #[allow(non_upper_case_globals)] // weird place to trigger this
        match value {
            whisper_gretype_WHISPER_GRETYPE_END => WhisperGrammarElementType::End,
            whisper_gretype_WHISPER_GRETYPE_ALT => WhisperGrammarElementType::Alternate,
            whisper_gretype_WHISPER_GRETYPE_RULE_REF => WhisperGrammarElementType::RuleReference,
            whisper_gretype_WHISPER_GRETYPE_CHAR => WhisperGrammarElementType::Character,
            whisper_gretype_WHISPER_GRETYPE_CHAR_NOT => WhisperGrammarElementType::NotCharacter,
            whisper_gretype_WHISPER_GRETYPE_CHAR_RNG_UPPER => {
                WhisperGrammarElementType::CharacterRangeUpper
            }
            whisper_gretype_WHISPER_GRETYPE_CHAR_ALT => {
                WhisperGrammarElementType::CharacterAlternate
            }
            _ => unreachable!(),
        }
    }
}

impl From<WhisperGrammarElementType> for whisper_rs_sys::whisper_gretype {
    fn from(value: WhisperGrammarElementType) -> Self {
        value as Self
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct WhisperGrammarElement {
    pub element_type: WhisperGrammarElementType,
    pub value: u32,
}

impl WhisperGrammarElement {
    /// Create a new grammar element
    /// 
    /// # Arguments
    /// * `element_type` - The type of grammar element
    /// * `value` - The value associated with the element
    /// 
    /// # Examples
    /// ```
    /// # use whisper_rs::{WhisperGrammarElement, WhisperGrammarElementType};
    /// let element = WhisperGrammarElement::new(WhisperGrammarElementType::Character, 65);
    /// assert_eq!(element.value, 65);
    /// ```
    pub fn new(element_type: WhisperGrammarElementType, value: u32) -> Self {
        use crate::common_logging::generic_debug;
        generic_debug!("Creating grammar element: type={}, value={}", element_type.to_string(), value);
        
        Self {
            element_type,
            value,
        }
    }

    /// Convert to C-compatible type
    /// 
    /// # Returns
    /// The corresponding C struct representation
    /// 
    /// # Examples
    /// ```
    /// # use whisper_rs::{WhisperGrammarElement, WhisperGrammarElementType};
    /// let element = WhisperGrammarElement::new(WhisperGrammarElementType::End, 0);
    /// let c_element = element.to_c_type();
    /// ```
    pub fn to_c_type(self) -> whisper_rs_sys::whisper_grammar_element {
        whisper_rs_sys::whisper_grammar_element {
            type_: self.element_type.into(),
            value: self.value,
        }
    }
    
    /// Create an end element
    /// 
    /// # Returns
    /// A grammar element representing the end of a rule
    pub fn end() -> Self {
        Self::new(WhisperGrammarElementType::End, 0)
    }
    
    /// Create a character element
    /// 
    /// # Arguments
    /// * `character` - The character code point
    /// 
    /// # Returns
    /// A grammar element representing a character
    pub fn character(character: u32) -> Self {
        Self::new(WhisperGrammarElementType::Character, character)
    }
    
    /// Create a rule reference element
    /// 
    /// # Arguments
    /// * `rule_id` - The ID of the rule to reference
    /// 
    /// # Returns
    /// A grammar element representing a rule reference
    pub fn rule_reference(rule_id: u32) -> Self {
        Self::new(WhisperGrammarElementType::RuleReference, rule_id)
    }
    
    /// Validate the element based on its type
    /// 
    /// # Returns
    /// `true` if the element is valid for its type, `false` otherwise
    pub fn is_valid(&self) -> bool {
        match self.element_type {
            WhisperGrammarElementType::End => self.value == 0,
            WhisperGrammarElementType::RuleReference => self.value > 0,
            WhisperGrammarElementType::Character => self.value <= 0x10FFFF, // Valid Unicode code point
            _ => true, // Other types can have any value
        }
    }
    
    /// Get a human-readable description of the element
    /// 
    /// # Returns
    /// String description of the element
    pub fn description(&self) -> String {
        match self.element_type {
            WhisperGrammarElementType::End => "End of rule".to_string(),
            WhisperGrammarElementType::Alternate => "Alternate rule".to_string(),
            WhisperGrammarElementType::RuleReference => format!("Reference to rule {}", self.value),
            WhisperGrammarElementType::Character => {
                if self.value <= 127 {
                    format!("Character '{}' ({})", self.value as u8 as char, self.value)
                } else {
                    format!("Character code {}", self.value)
                }
            },
            WhisperGrammarElementType::NotCharacter => format!("Not character {}", self.value),
            WhisperGrammarElementType::CharacterRangeUpper => format!("Range upper bound {}", self.value),
            WhisperGrammarElementType::CharacterAlternate => format!("Alternate character {}", self.value),
        }
    }
}

impl std::fmt::Display for WhisperGrammarElement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({})", self.element_type.to_string(), self.value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_grammar_element_type_string_conversion() {
        assert_eq!(WhisperGrammarElementType::End.to_string(), "End");
        assert_eq!(WhisperGrammarElementType::Alternate.to_string(), "Alternate");
        assert_eq!(WhisperGrammarElementType::RuleReference.to_string(), "RuleReference");
        assert_eq!(WhisperGrammarElementType::Character.to_string(), "Character");
        assert_eq!(WhisperGrammarElementType::NotCharacter.to_string(), "NotCharacter");
        assert_eq!(WhisperGrammarElementType::CharacterRangeUpper.to_string(), "CharacterRangeUpper");
        assert_eq!(WhisperGrammarElementType::CharacterAlternate.to_string(), "CharacterAlternate");
    }
    
    #[test]
    fn test_grammar_element_type_categorization() {
        // Test terminal elements
        assert!(WhisperGrammarElementType::Character.is_terminal());
        assert!(WhisperGrammarElementType::NotCharacter.is_terminal());
        assert!(WhisperGrammarElementType::CharacterRangeUpper.is_terminal());
        assert!(WhisperGrammarElementType::CharacterAlternate.is_terminal());
        
        // Test control elements
        assert!(WhisperGrammarElementType::End.is_control());
        assert!(WhisperGrammarElementType::Alternate.is_control());
        assert!(WhisperGrammarElementType::RuleReference.is_control());
        
        // Test mutual exclusivity
        for variant in [
            WhisperGrammarElementType::End,
            WhisperGrammarElementType::Alternate,
            WhisperGrammarElementType::RuleReference,
            WhisperGrammarElementType::Character,
            WhisperGrammarElementType::NotCharacter,
            WhisperGrammarElementType::CharacterRangeUpper,
            WhisperGrammarElementType::CharacterAlternate,
        ] {
            assert!(variant.is_terminal() != variant.is_control());
        }
    }
    
    #[test]
    fn test_raw_value_validation() {
        // Valid values (0-6)
        for i in 0..=6 {
            assert!(WhisperGrammarElementType::is_valid_raw(i));
        }
        
        // Invalid values
        assert!(!WhisperGrammarElementType::is_valid_raw(7));
        assert!(!WhisperGrammarElementType::is_valid_raw(100));
        assert!(!WhisperGrammarElementType::is_valid_raw(u32::MAX));
    }
    
    #[test]
    fn test_grammar_element_type_conversions() {
        let types = [
            (WhisperGrammarElementType::End, whisper_gretype_WHISPER_GRETYPE_END),
            (WhisperGrammarElementType::Alternate, whisper_gretype_WHISPER_GRETYPE_ALT),
            (WhisperGrammarElementType::RuleReference, whisper_gretype_WHISPER_GRETYPE_RULE_REF),
            (WhisperGrammarElementType::Character, whisper_gretype_WHISPER_GRETYPE_CHAR),
            (WhisperGrammarElementType::NotCharacter, whisper_gretype_WHISPER_GRETYPE_CHAR_NOT),
            (WhisperGrammarElementType::CharacterRangeUpper, whisper_gretype_WHISPER_GRETYPE_CHAR_RNG_UPPER),
            (WhisperGrammarElementType::CharacterAlternate, whisper_gretype_WHISPER_GRETYPE_CHAR_ALT),
        ];
        
        for (element_type, raw_value) in types {
            // Test conversion to raw
            let converted_raw: whisper_rs_sys::whisper_gretype = element_type.into();
            assert_eq!(converted_raw, raw_value);
            
            // Test conversion from raw
            let converted_back = WhisperGrammarElementType::from(raw_value);
            assert_eq!(converted_back, element_type);
        }
    }
    
    #[test]
    #[should_panic(expected = "Invalid WhisperGrammarElementType value")]
    fn test_invalid_conversion_panics() {
        let _ = WhisperGrammarElementType::from(999);
    }
    
    #[test]
    fn test_grammar_element_creation() {
        let element = WhisperGrammarElement::new(WhisperGrammarElementType::Character, 65);
        assert_eq!(element.element_type, WhisperGrammarElementType::Character);
        assert_eq!(element.value, 65);
    }
    
    #[test]
    fn test_grammar_element_factory_methods() {
        let end_element = WhisperGrammarElement::end();
        assert_eq!(end_element.element_type, WhisperGrammarElementType::End);
        assert_eq!(end_element.value, 0);
        
        let char_element = WhisperGrammarElement::character(97);
        assert_eq!(char_element.element_type, WhisperGrammarElementType::Character);
        assert_eq!(char_element.value, 97);
        
        let rule_element = WhisperGrammarElement::rule_reference(5);
        assert_eq!(rule_element.element_type, WhisperGrammarElementType::RuleReference);
        assert_eq!(rule_element.value, 5);
    }
    
    #[test]
    fn test_grammar_element_validation() {
        // Valid elements
        assert!(WhisperGrammarElement::end().is_valid());
        assert!(WhisperGrammarElement::character(65).is_valid());
        assert!(WhisperGrammarElement::rule_reference(1).is_valid());
        
        // Invalid elements
        let invalid_end = WhisperGrammarElement::new(WhisperGrammarElementType::End, 1);
        assert!(!invalid_end.is_valid());
        
        let invalid_rule = WhisperGrammarElement::new(WhisperGrammarElementType::RuleReference, 0);
        assert!(!invalid_rule.is_valid());
        
        let invalid_char = WhisperGrammarElement::new(WhisperGrammarElementType::Character, 0x110000);
        assert!(!invalid_char.is_valid());
    }
    
    #[test]
    fn test_grammar_element_description() {
        let end_element = WhisperGrammarElement::end();
        assert_eq!(end_element.description(), "End of rule");
        
        let char_element = WhisperGrammarElement::character(65);
        assert!(char_element.description().contains("Character 'A'"));
        
        let high_char = WhisperGrammarElement::character(0x1F600);
        assert!(high_char.description().contains("Character code"));
        
        let rule_element = WhisperGrammarElement::rule_reference(5);
        assert_eq!(rule_element.description(), "Reference to rule 5");
    }
    
    #[test]
    fn test_grammar_element_display() {
        let element = WhisperGrammarElement::character(65);
        let display_str = format!("{}", element);
        assert_eq!(display_str, "Character(65)");
    }
    
    #[test]
    fn test_grammar_element_c_conversion() {
        let element = WhisperGrammarElement::character(65);
        let c_element = element.to_c_type();
        
        assert_eq!(c_element.type_, whisper_gretype_WHISPER_GRETYPE_CHAR);
        assert_eq!(c_element.value, 65);
    }
    
    #[test]
    fn test_grammar_element_traits() {
        let element1 = WhisperGrammarElement::character(65);
        let element2 = WhisperGrammarElement::character(65);
        let element3 = WhisperGrammarElement::character(66);
        
        // Test PartialEq
        assert_eq!(element1, element2);
        assert_ne!(element1, element3);
        
        // Test Clone and Copy
        let cloned = element1.clone();
        let copied = element1;
        assert_eq!(element1, cloned);
        assert_eq!(element1, copied);
        
        // Test Debug
        let debug_str = format!("{:?}", element1);
        assert!(debug_str.contains("Character"));
        assert!(debug_str.contains("65"));
        
        // Test Hash (just ensure it doesn't panic)
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(element1);
        set.insert(element2);
        set.insert(element3);
        assert_eq!(set.len(), 2); // element1 and element2 are equal
    }
    
    #[test]
    fn test_unicode_boundary_conditions() {
        // Test maximum valid Unicode code point
        let max_unicode = WhisperGrammarElement::character(0x10FFFF);
        assert!(max_unicode.is_valid());
        
        // Test just over the maximum
        let invalid_unicode = WhisperGrammarElement::character(0x110000);
        assert!(!invalid_unicode.is_valid());
        
        // Test common ASCII characters
        let ascii_chars = [0, 32, 65, 97, 127];
        for ch in ascii_chars {
            let element = WhisperGrammarElement::character(ch);
            assert!(element.is_valid());
        }
    }
}
