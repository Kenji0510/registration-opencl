use anyhow::Result;
use ndarray::Array2;
use ocl::{Buffer};


pub fn convert_dtoh(buffer: &Buffer<f32>, pts_num: usize) -> Result<Array2<f32>> {
    let buffer_len = buffer.len();
    let expected_len = pts_num * 3;
    
    // バッファサイズが期待値より小さい場合はエラー
    if buffer_len < expected_len {
        eprintln!(
            "Buffer too small: buffer has {} elements, but {} needed (pts_num={} * 3)",
            buffer_len, expected_len, pts_num
        );
        return Err(anyhow::anyhow!(
            "Buffer too small: buffer has {} elements, but {} needed (pts_num={} * 3)",
            buffer_len, expected_len, pts_num
        ));
    }

    let mut vec_pts = vec![f32::default(); pts_num * 3];
    buffer.read(&mut vec_pts).enq()?;

    let mut arr = Array2::<f32>::zeros((pts_num, 3));
    for i in 0..pts_num {
        arr[[i, 0]] = vec_pts[i * 3];
        arr[[i, 1]] = vec_pts[i * 3 + 1];
        arr[[i, 2]] = vec_pts[i * 3 + 2];
    }

    Ok(arr)
}