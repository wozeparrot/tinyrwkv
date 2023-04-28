use std::ffi::CStr;
use std::path::PathBuf;

use tokenizers::tokenizer::Tokenizer;

#[no_mangle]
pub extern "C" fn tk_from_file(file: *const libc::c_char) -> *mut libc::c_void {
    let file_cstr = unsafe { CStr::from_ptr(file) };
    let file_str = file_cstr.to_str().unwrap();
    let file_path = PathBuf::from(file_str);

    match Tokenizer::from_file(file_path) {
        Ok(tokenizer) => Box::into_raw(Box::new(tokenizer)) as *mut libc::c_void,
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn tk_free(tokenizer: *mut libc::c_void) {
    if !tokenizer.is_null() {
        unsafe { Box::from_raw(tokenizer as *mut Tokenizer) };
    }
}

#[no_mangle]
pub extern "C" fn tk_encode(
    tokenizer: *mut libc::c_void,
    text: *const libc::c_char,
    len: *mut u32,
) -> *mut u32 {
    let tokenizer = unsafe { &*(tokenizer as *mut Tokenizer) };
    let text_cstr = unsafe { CStr::from_ptr(text) };
    let text_str = text_cstr.to_string_lossy();

    let tokens = tokenizer.encode(text_str, false).unwrap();
    let ids = tokens.get_ids();

    let mut ids_vec = ids.to_vec();
    ids_vec.shrink_to_fit();

    let ids_ptr = ids_vec.as_mut_ptr();
    unsafe { *len = ids_vec.len() as u32 };
    std::mem::forget(ids_vec);

    ids_ptr
}

#[no_mangle]
pub extern "C" fn tk_decode(
    tokenizer: *mut libc::c_void,
    tokens: *const u32,
    len: u32,
) -> *mut libc::c_char {
    let tokenizer = unsafe { &*(tokenizer as *mut Tokenizer) };
    let tokens_slice = unsafe { std::slice::from_raw_parts(tokens, len as usize) };

    let text = tokenizer.decode(tokens_slice.to_vec(), false).unwrap();

    let text_cstr = std::ffi::CString::new(text).unwrap();
    let text_ptr = text_cstr.into_raw();

    text_ptr
}
