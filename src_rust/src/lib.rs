use ahash::AHashMap;
use arrow_array::{
    Array, Float32Array, Float64Array, Int32Array, Int64Array, UInt32Array, UInt64Array,
};
use arrow_schema::DataType;
use chrono::{DateTime, Datelike, TimeZone, Utc};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString};
use pyo3_arrow::PyTable;
use rayon::prelude::*;
use regex::Regex;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Arrow helpers
// ---------------------------------------------------------------------------

fn numeric_value(array: &dyn Array, index: usize) -> PyResult<Option<f64>> {
    if array.is_null(index) {
        return Ok(None);
    }
    let value = match array.data_type() {
        DataType::Float64 => array
            .as_any()
            .downcast_ref::<Float64Array>()
            .map(|values| values.value(index)),
        DataType::Float32 => array
            .as_any()
            .downcast_ref::<Float32Array>()
            .map(|values| values.value(index) as f64),
        DataType::Int64 => array
            .as_any()
            .downcast_ref::<Int64Array>()
            .map(|values| values.value(index) as f64),
        DataType::Int32 => array
            .as_any()
            .downcast_ref::<Int32Array>()
            .map(|values| values.value(index) as f64),
        DataType::UInt64 => array
            .as_any()
            .downcast_ref::<UInt64Array>()
            .map(|values| values.value(index) as f64),
        DataType::UInt32 => array
            .as_any()
            .downcast_ref::<UInt32Array>()
            .map(|values| values.value(index) as f64),
        _ => None,
    };
    value.map(Some).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
            "Arrow column dtype {:?} is not a supported numeric dtype",
            array.data_type()
        ))
    })
}

fn for_each_numeric(
    table: &PyTable,
    column_name: &str,
    mut operation: impl FnMut(Option<f64>),
) -> PyResult<()> {
    let batches = table.batches();
    if batches.is_empty() {
        return Ok(());
    }
    for batch in batches {
        let column_index = batch.schema().index_of(column_name).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "Arrow table has no column named {column_name:?}"
            ))
        })?;
        let array = batch.column(column_index);
        for index in 0..array.len() {
            operation(numeric_value(array.as_ref(), index)?);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Arrow comparison + numeric window
// ---------------------------------------------------------------------------

#[pyfunction]
fn filter_compare_arrow<'py>(
    py: Python<'py>,
    table: PyTable,
    column_name: &str,
    op: &str,
    scalar: f64,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let mut values = Vec::new();
    for_each_numeric(&table, column_name, |value| values.push(value))?;
    let mask = py.allow_threads(|| {
        values
            .par_iter()
            .map(|value| match value {
                Some(value) if !value.is_nan() && !scalar.is_nan() => match op {
                    ">" => *value > scalar,
                    "<" => *value < scalar,
                    ">=" => *value >= scalar,
                    "<=" => *value <= scalar,
                    "==" => (*value - scalar).abs() < f64::EPSILON,
                    "!=" => (*value - scalar).abs() >= f64::EPSILON,
                    _ => false,
                },
                _ => false,
            })
            .collect::<Vec<_>>()
    });
    Ok(PyArray1::from_vec_bound(py, mask))
}

#[pyfunction]
fn numeric_window_arrow<'py>(
    py: Python<'py>,
    table: PyTable,
    column_name: &str,
    operation: &str,
) -> PyResult<PyObject> {
    let mut arrow_values = Vec::new();
    for_each_numeric(&table, column_name, |value| arrow_values.push(value))?;
    let result = py.allow_threads(|| match operation {
        "var" | "std" => {
            let (count, _, sum_squared_differences) = arrow_values
                .par_iter()
                .filter_map(|value| value.filter(|value| !value.is_nan()))
                .map(|value| (1usize, value, 0.0))
                .reduce(
                    || (0usize, 0.0, 0.0),
                    |(left_count, left_mean, left_m2), (right_count, right_mean, right_m2)| {
                        if left_count == 0 {
                            return (right_count, right_mean, right_m2);
                        }
                        if right_count == 0 {
                            return (left_count, left_mean, left_m2);
                        }
                        let count = left_count + right_count;
                        let delta = right_mean - left_mean;
                        let mean = left_mean + delta * right_count as f64 / count as f64;
                        let m2 = left_m2
                            + right_m2
                            + delta.powi(2) * left_count as f64 * right_count as f64 / count as f64;
                        (count, mean, m2)
                    },
                );
            if count <= 1 {
                return NumericWindowResult::Scalar(f64::NAN);
            }
            let variance = sum_squared_differences / (count - 1) as f64;
            NumericWindowResult::Scalar(if operation == "std" {
                variance.sqrt()
            } else {
                variance
            })
        }
        "cumsum" | "cummax" | "cummin" => {
            const CHUNK_SIZE: usize = 8192;
            let chunks = arrow_values
                .par_chunks(CHUNK_SIZE)
                .map(|chunk| {
                    let mut current = match operation {
                        "cummax" => f64::NEG_INFINITY,
                        "cummin" => f64::INFINITY,
                        _ => 0.0,
                    };
                    let mut prefix = Vec::with_capacity(chunk.len());
                    for value in chunk {
                        if let Some(value) = value.filter(|value| !value.is_nan()) {
                            match operation {
                                "cumsum" => current += value,
                                "cummax" if value > current => current = value,
                                "cummin" if value < current => current = value,
                                _ => {}
                            }
                            prefix.push(current);
                        } else {
                            prefix.push(f64::NAN);
                        }
                    }
                    (prefix, current)
                })
                .collect::<Vec<_>>();
            let mut offsets = Vec::with_capacity(chunks.len());
            let mut accumulated = match operation {
                "cummax" => f64::NEG_INFINITY,
                "cummin" => f64::INFINITY,
                _ => 0.0,
            };
            for (_, chunk_value) in &chunks {
                offsets.push(accumulated);
                match operation {
                    "cumsum" => accumulated += chunk_value,
                    "cummax" => accumulated = accumulated.max(*chunk_value),
                    "cummin" => accumulated = accumulated.min(*chunk_value),
                    _ => {}
                }
            }
            let output = chunks
                .into_par_iter()
                .zip(offsets.into_par_iter())
                .flat_map_iter(|((prefix, _), offset)| {
                    prefix.into_iter().map(move |value| {
                        if value.is_nan() {
                            f64::NAN
                        } else {
                            match operation {
                                "cumsum" => value + offset,
                                "cummax" => value.max(offset),
                                "cummin" => value.min(offset),
                                _ => value,
                            }
                        }
                    })
                })
                .collect::<Vec<_>>();
            debug_assert_eq!(
                output.len(),
                arrow_values.len(),
                "chunked cumulative output length mismatch"
            );
            if output.len() != arrow_values.len() {
                return NumericWindowResult::InternalError;
            }
            NumericWindowResult::Values(output)
        }
        _ => NumericWindowResult::Unsupported,
    });
    match result {
        NumericWindowResult::Scalar(value) => Ok(value.into_py(py)),
        NumericWindowResult::Values(values) => Ok(PyArray1::from_vec_bound(py, values).into_py(py)),
        NumericWindowResult::Unsupported => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Unsupported Arrow numeric operation: {operation:?}"),
        )),
        NumericWindowResult::InternalError => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "internal error: chunked cumulative output length mismatch",
        )),
    }
}

enum NumericWindowResult {
    Scalar(f64),
    Values(Vec<f64>),
    Unsupported,
    InternalError,
}

// ---------------------------------------------------------------------------
// NumPy array comparisons / masked reductions
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (arr, op, scalar=None, other_arr=None))]
fn filter_compare<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, f64>,
    op: &str,
    scalar: Option<f64>,
    other_arr: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let slice = arr.as_slice().unwrap();
    let other_slice = other_arr.map(|a| a.as_slice().unwrap().to_vec());

    let mask: Vec<bool> = py.allow_threads(|| {
        slice
            .par_iter()
            .enumerate()
            .map(|(i, &v)| {
                if v.is_nan() {
                    return false;
                }
                let target = if let Some(other) = &other_slice {
                    if i < other.len() {
                        other[i]
                    } else {
                        return false;
                    }
                } else if let Some(s) = scalar {
                    s
                } else {
                    return false;
                };
                if target.is_nan() {
                    return false;
                }
                match op {
                    ">" => v > target,
                    "<" => v < target,
                    ">=" => v >= target,
                    "<=" => v <= target,
                    "==" => (v - target).abs() < f64::EPSILON,
                    "!=" => (v - target).abs() >= f64::EPSILON,
                    _ => false,
                }
            })
            .collect()
    });
    Ok(PyArray1::from_vec_bound(py, mask))
}

#[pyfunction]
#[pyo3(signature = (arr, mask=None))]
fn sum_masked<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, f64>,
    mask: Option<PyReadonlyArray1<'py, bool>>,
) -> f64 {
    let slice = arr.as_slice().unwrap();
    let mask_slice = mask.map(|m| m.as_slice().unwrap().to_vec());
    py.allow_threads(|| match mask_slice {
        Some(m) => slice
            .par_iter()
            .zip(m.par_iter())
            .filter(|&(&v, &keep)| keep && !v.is_nan())
            .map(|(&v, _)| v)
            .sum(),
        None => slice.par_iter().filter(|&&v| !v.is_nan()).sum(),
    })
}

#[pyfunction]
#[pyo3(signature = (arr, mask=None))]
fn mean_masked<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, f64>,
    mask: Option<PyReadonlyArray1<'py, bool>>,
) -> f64 {
    let slice = arr.as_slice().unwrap();
    let mask_slice = mask.map(|m| m.as_slice().unwrap().to_vec());
    let (sum, count) = py.allow_threads(|| match mask_slice {
        Some(m) => {
            let valid: Vec<f64> = slice
                .par_iter()
                .zip(m.par_iter())
                .filter(|&(&v, &keep)| keep && !v.is_nan())
                .map(|(&v, _)| v)
                .collect();
            (valid.par_iter().sum::<f64>(), valid.len())
        }
        None => {
            let valid: Vec<f64> = slice
                .par_iter()
                .filter(|&&v| !v.is_nan())
                .copied()
                .collect();
            (valid.par_iter().sum::<f64>(), valid.len())
        }
    });
    if count == 0 {
        f64::NAN
    } else {
        sum / (count as f64)
    }
}

#[pyfunction]
fn var_parallel<'py>(py: Python<'py>, arr: PyReadonlyArray1<'py, f64>) -> f64 {
    let slice = arr.as_slice().unwrap();
    let (mean, count) = py.allow_threads(|| {
        let valid: Vec<f64> = slice
            .par_iter()
            .filter(|&&v| !v.is_nan())
            .copied()
            .collect();
        (
            valid.par_iter().sum::<f64>() / valid.len() as f64,
            valid.len(),
        )
    });
    if count <= 1 {
        return f64::NAN;
    }
    py.allow_threads(|| {
        let sum_sq_diff: f64 = slice
            .par_iter()
            .filter(|&&v| !v.is_nan())
            .map(|&v| (v - mean).powi(2))
            .sum();
        sum_sq_diff / (count - 1) as f64
    })
}

#[pyfunction]
fn std_parallel<'py>(py: Python<'py>, arr: PyReadonlyArray1<'py, f64>) -> f64 {
    var_parallel(py, arr).sqrt()
}

#[pyfunction]
fn corr_parallel<'py>(
    py: Python<'py>,
    arr1: PyReadonlyArray1<'py, f64>,
    arr2: PyReadonlyArray1<'py, f64>,
) -> f64 {
    let s1 = arr1.as_slice().unwrap();
    let s2 = arr2.as_slice().unwrap();
    if s1.len() != s2.len() {
        return f64::NAN;
    }

    let (n, sum1, sum2) = py.allow_threads(|| {
        s1.par_iter()
            .zip(s2.par_iter())
            .filter(|&(&a, &b)| !a.is_nan() && !b.is_nan())
            .fold(
                || (0usize, 0.0f64, 0.0f64),
                |(n, s1, s2), (&a, &b)| (n + 1, s1 + a, s2 + b),
            )
            .reduce(
                || (0usize, 0.0f64, 0.0f64),
                |(n1, s1a, s2a), (n2, s1b, s2b)| (n1 + n2, s1a + s1b, s2a + s2b),
            )
    });

    if n <= 1 {
        return f64::NAN;
    }

    let mean1 = sum1 / n as f64;
    let mean2 = sum2 / n as f64;

    let (num, d1, d2) = py.allow_threads(|| {
        s1.par_iter()
            .zip(s2.par_iter())
            .filter(|&(&a, &b)| !a.is_nan() && !b.is_nan())
            .fold(
                || (0.0f64, 0.0f64, 0.0f64),
                |(num, d1, d2), (&a, &b)| {
                    let da = a - mean1;
                    let db = b - mean2;
                    (num + da * db, d1 + da * da, d2 + db * db)
                },
            )
            .reduce(
                || (0.0f64, 0.0f64, 0.0f64),
                |(n1, d1a, d2a), (n2, d1b, d2b)| (n1 + n2, d1a + d1b, d2a + d2b),
            )
    });

    let denom = (d1 * d2).sqrt();
    if denom == 0.0 {
        return f64::NAN;
    }
    num / denom
}

#[pyfunction]
fn cov_parallel<'py>(
    py: Python<'py>,
    arr1: PyReadonlyArray1<'py, f64>,
    arr2: PyReadonlyArray1<'py, f64>,
) -> f64 {
    let s1 = arr1.as_slice().unwrap();
    let s2 = arr2.as_slice().unwrap();
    if s1.len() != s2.len() {
        return f64::NAN;
    }

    let (n, sum1, sum2) = py.allow_threads(|| {
        s1.par_iter()
            .zip(s2.par_iter())
            .filter(|&(&a, &b)| !a.is_nan() && !b.is_nan())
            .fold(
                || (0usize, 0.0f64, 0.0f64),
                |(n, s1, s2), (&a, &b)| (n + 1, s1 + a, s2 + b),
            )
            .reduce(
                || (0usize, 0.0f64, 0.0f64),
                |(n1, s1a, s2a), (n2, s1b, s2b)| (n1 + n2, s1a + s1b, s2a + s2b),
            )
    });

    if n <= 1 {
        return f64::NAN;
    }

    let mean1 = sum1 / n as f64;
    let mean2 = sum2 / n as f64;

    let sum = py.allow_threads(|| {
        s1.par_iter()
            .zip(s2.par_iter())
            .filter(|&(&a, &b)| !a.is_nan() && !b.is_nan())
            .map(|(&a, &b)| (a - mean1) * (b - mean2))
            .sum::<f64>()
    });

    sum / (n - 1) as f64
}

// ---------------------------------------------------------------------------
// Cumulative operations (sequential — kept for API compatibility)
// ---------------------------------------------------------------------------

#[pyfunction]
fn cumsum_parallel<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    let slice = arr.as_slice().unwrap();
    let mut result = Vec::with_capacity(slice.len());
    let mut current = 0.0;
    for &v in slice {
        if !v.is_nan() {
            current += v;
        }
        result.push(current);
    }
    PyArray1::from_vec_bound(py, result)
}

#[pyfunction]
fn cummax_parallel<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    let slice = arr.as_slice().unwrap();
    let mut result = Vec::with_capacity(slice.len());
    let mut current = f64::NEG_INFINITY;
    for &v in slice {
        if !v.is_nan() && v > current {
            current = v;
        }
        result.push(current);
    }
    PyArray1::from_vec_bound(py, result)
}

#[pyfunction]
fn cummin_parallel<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    let slice = arr.as_slice().unwrap();
    let mut result = Vec::with_capacity(slice.len());
    let mut current = f64::INFINITY;
    for &v in slice {
        if !v.is_nan() && v < current {
            current = v;
        }
        result.push(current);
    }
    PyArray1::from_vec_bound(py, result)
}

// ---------------------------------------------------------------------------
// Scalar math / sorting
// ---------------------------------------------------------------------------

#[pyfunction]
fn math_scalar<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, f64>,
    op: &str,
    scalar: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice = arr.as_slice().unwrap();
    let result: Vec<f64> = py.allow_threads(|| {
        slice
            .par_iter()
            .map(|&v| match op {
                "+" => Ok(v + scalar),
                "-" => Ok(v - scalar),
                "*" => Ok(v * scalar),
                "/" => Ok(v / scalar),
                _ => Err(()),
            })
            .collect::<Result<Vec<_>, _>>()
    })
    .map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unsupported scalar operation: {op:?}"
        ))
    })?;
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
fn sort_parallel<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, f64>,
    descending: bool,
) -> Bound<'py, PyArray1<f64>> {
    let mut slice = arr.as_slice().unwrap().to_vec();
    py.allow_threads(|| {
        slice.par_sort_unstable_by(|a, b| {
            if descending {
                b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            }
        });
    });
    PyArray1::from_vec_bound(py, slice)
}

#[pyfunction]
fn nlargest_parallel<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, f64>,
    n: usize,
) -> Bound<'py, PyArray1<f64>> {
    let slice = arr.as_slice().unwrap();
    let mut vec = slice.to_vec();
    py.allow_threads(|| {
        vec.par_sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        vec.truncate(n);
    });
    PyArray1::from_vec_bound(py, vec)
}

#[pyfunction]
fn nsmallest_parallel<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, f64>,
    n: usize,
) -> Bound<'py, PyArray1<f64>> {
    let slice = arr.as_slice().unwrap();
    let mut vec = slice.to_vec();
    py.allow_threads(|| {
        vec.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        vec.truncate(n);
    });
    PyArray1::from_vec_bound(py, vec)
}

// ---------------------------------------------------------------------------
// unique + value_counts take PyReadonlyArray1 (true zero-copy input)
// ---------------------------------------------------------------------------

#[pyfunction]
fn unique_parallel<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let slice = arr.as_slice().unwrap();
    let result: Vec<f64> = py.allow_threads(|| {
        use std::collections::HashSet;
        let mut seen: HashSet<u64> = HashSet::with_capacity(slice.len());
        slice
            .iter()
            .copied()
            .filter(|v| seen.insert(v.to_bits()))
            .collect()
    });
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
fn value_counts_parallel<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i64>>)> {
    let slice = arr.as_slice().unwrap();
    let (values, counts): (Vec<f64>, Vec<i64>) = py.allow_threads(|| {
        let mut map: HashMap<u64, usize> = HashMap::with_capacity(slice.len());
        for v in slice {
            *map.entry(v.to_bits()).or_insert(0) += 1;
        }
        let mut pairs: Vec<(f64, i64)> = map
            .into_iter()
            .map(|(bits, count)| (f64::from_bits(bits), count as i64))
            .collect();
        pairs.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        pairs.into_iter().unzip()
    });
    Ok((
        PyArray1::from_vec_bound(py, values),
        PyArray1::from_vec_bound(py, counts),
    ))
}

// ---------------------------------------------------------------------------
// text_* accept PyReadonlyArray1<PyObject> (no Python list round-trip)
// ---------------------------------------------------------------------------

fn collect_strings<'py>(
    py: Python<'py>,
    arr: &PyReadonlyArray1<'py, PyObject>,
) -> PyResult<Vec<String>> {
    let slice = arr.as_slice()?;
    let mut out = Vec::with_capacity(slice.len());
    for obj in slice {
        let s = obj.bind(py).downcast::<PyString>()?;
        out.push(s.to_string());
    }
    Ok(out)
}

fn strings_to_pyarray<'py>(
    py: Python<'py>,
    values: Vec<String>,
) -> Bound<'py, PyArray1<PyObject>> {
    let py_objs: Vec<PyObject> = values
        .into_iter()
        .map(|s| PyString::new_bound(py, &s).into())
        .collect();
    PyArray1::from_vec_bound(py, py_objs)
}

#[pyfunction]
fn text_lower_parallel<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, PyObject>,
) -> PyResult<Bound<'py, PyArray1<PyObject>>> {
    let strings = collect_strings(py, &arr)?;
    let result: Vec<String> =
        py.allow_threads(|| strings.into_par_iter().map(|s| s.to_lowercase()).collect());
    Ok(strings_to_pyarray(py, result))
}

#[pyfunction]
fn text_upper_parallel<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, PyObject>,
) -> PyResult<Bound<'py, PyArray1<PyObject>>> {
    let strings = collect_strings(py, &arr)?;
    let result: Vec<String> =
        py.allow_threads(|| strings.into_par_iter().map(|s| s.to_uppercase()).collect());
    Ok(strings_to_pyarray(py, result))
}

#[pyfunction]
fn text_contains_parallel<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, PyObject>,
    pat: String,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let strings = collect_strings(py, &arr)?;
    let re = Regex::new(&pat)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let result: Vec<bool> =
        py.allow_threads(|| strings.into_par_iter().map(|s| re.is_match(&s)).collect());
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
fn text_len_parallel<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, PyObject>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let strings = collect_strings(py, &arr)?;
    let result: Vec<i64> =
        py.allow_threads(|| strings.into_par_iter().map(|s| s.len() as i64).collect());
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
fn text_replace_parallel<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, PyObject>,
    pat: String,
    repl: String,
) -> PyResult<Bound<'py, PyArray1<PyObject>>> {
    let strings = collect_strings(py, &arr)?;
    let re = Regex::new(&pat)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let result: Vec<String> = py.allow_threads(|| {
        strings
            .into_par_iter()
            .map(|s| re.replace_all(&s, repl.as_str()).to_string())
            .collect()
    });
    Ok(strings_to_pyarray(py, result))
}

#[pyfunction]
fn text_strip_parallel<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, PyObject>,
) -> PyResult<Bound<'py, PyArray1<PyObject>>> {
    let strings = collect_strings(py, &arr)?;
    let result: Vec<String> =
        py.allow_threads(|| strings.into_par_iter().map(|s| s.trim().to_string()).collect());
    Ok(strings_to_pyarray(py, result))
}

// ---------------------------------------------------------------------------
// groupby_sum uses AHashMap<&str, f64> (no per-key String clone)
// ---------------------------------------------------------------------------

#[pyfunction]
fn groupby_sum_parallel<'py>(
    py: Python<'py>,
    keys: Vec<String>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let vals = values.as_slice().unwrap();
    if keys.len() != vals.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Length mismatch",
        ));
    }

    let map: AHashMap<&str, f64> = py.allow_threads(|| {
        let mut m = AHashMap::with_capacity(keys.len());
        for (k, &v) in keys.iter().zip(vals.iter()) {
            if !v.is_nan() {
                *m.entry(k.as_str()).or_insert(0.0) += v;
            }
        }
        m
    });

    let mut result: Vec<f64> = map.into_values().collect();
    result.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(PyArray1::from_vec_bound(py, result))
}

// ---------------------------------------------------------------------------
// Datetime helpers
// ---------------------------------------------------------------------------

#[pyfunction]
fn parse_datetime_parallel(py: Python<'_>, arr: Vec<String>, fmt: &str) -> PyResult<Py<PyAny>> {
    let result: Vec<i64> = py.allow_threads(|| {
        arr.into_par_iter()
            .filter_map(|s| {
                DateTime::parse_from_str(&s, fmt)
                    .ok()
                    .map(|dt| dt.timestamp())
            })
            .collect()
    });
    let list = PyList::new_bound(py, result);
    Ok(list.into())
}

#[pyfunction]
fn dt_year_parallel(py: Python<'_>, arr: Vec<i64>) -> PyResult<Py<PyAny>> {
    let result: Vec<i32> = py.allow_threads(|| {
        arr.into_par_iter()
            .filter_map(|ts| Utc.timestamp_opt(ts, 0).single().map(|dt| dt.year()))
            .collect()
    });
    let list = PyList::new_bound(py, result);
    Ok(list.into())
}

#[pyfunction]
fn dt_month_parallel(py: Python<'_>, arr: Vec<i64>) -> PyResult<Py<PyAny>> {
    let result: Vec<u32> = py.allow_threads(|| {
        arr.into_par_iter()
            .filter_map(|ts| Utc.timestamp_opt(ts, 0).single().map(|dt| dt.month()))
            .collect()
    });
    let list = PyList::new_bound(py, result);
    Ok(list.into())
}

#[pyfunction]
fn dt_day_parallel(py: Python<'_>, arr: Vec<i64>) -> PyResult<Py<PyAny>> {
    let result: Vec<u32> = py.allow_threads(|| {
        arr.into_par_iter()
            .filter_map(|ts| Utc.timestamp_opt(ts, 0).single().map(|dt| dt.day()))
            .collect()
    });
    let list = PyList::new_bound(py, result);
    Ok(list.into())
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
fn dfmacro_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(filter_compare, m)?)?;
    m.add_function(wrap_pyfunction!(filter_compare_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(numeric_window_arrow, m)?)?;
    m.add_function(wrap_pyfunction!(sum_masked, m)?)?;
    m.add_function(wrap_pyfunction!(mean_masked, m)?)?;
    m.add_function(wrap_pyfunction!(var_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(std_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(corr_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(cov_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(cumsum_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(cummax_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(cummin_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(math_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(sort_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(nlargest_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(nsmallest_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(unique_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(value_counts_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(text_lower_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(text_upper_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(text_contains_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(text_len_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(text_replace_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(text_strip_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(groupby_sum_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(parse_datetime_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(dt_year_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(dt_month_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(dt_day_parallel, m)?)?;
    Ok(())
}
