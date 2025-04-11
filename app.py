from flask import Flask, request, jsonify, render_template
import numpy as np
import math

app = Flask(__name__)

def validate_numeric_input(input_str, min_length=1):
    """Validates and converts comma-separated input string to list of numbers"""
    if not input_str:
        raise ValueError("Input cannot be empty")
    try:
        values = [x.strip() for x in input_str.split(',')]
        result = []
        for x in values:
            try:
                val = int(x)
                if float(x) == val:
                    result.append(val)  
                else:
                    result.append(float(x))  
            except ValueError:
                result.append(float(x)) 
        return result
    except ValueError:
        raise ValueError("All inputs must be numeric")

def validate_matrix_input(input_str):
    """Validates and converts semicolon-separated matrix input to numpy array"""
    if not input_str:
        raise ValueError("Matrix input cannot be empty")
    rows = [r.split(',') for r in input_str.strip().split(';')]
    if not rows or any(not row for row in rows):
        raise ValueError("Invalid matrix format")
    return np.array([[float(x.strip()) for x in row] for row in rows])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/exam1', methods=['POST'])
def exam1():
    """Finds the second largest number in a list"""
    try:
        data = request.json
        if data.get('example'):
            numbers = [10, 5, 8, 20, 15]
        else:
            numbers = validate_numeric_input(data.get('input', ''), min_length=2)
        if len(numbers) < 2:
            raise ValueError("List must have at least 2 numbers")
        largest = max(numbers[0], numbers[1])
        second_largest = min(numbers[0], numbers[1])
        for i in range(2, len(numbers)):
            if numbers[i] > largest:
                second_largest = largest
                largest = numbers[i]
            elif numbers[i] > second_largest and numbers[i] != largest:
                second_largest = numbers[i]
        return jsonify({'success': True, 'result': {'second_largest': second_largest}})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/exam2', methods=['POST'])
def exam2():
    """Finds the smallest number with highest frequency in a list"""
    try:
        data = request.json
        if data.get('example'):
            lst = [1, 3, 3, 2, 1, 3, 4]
        else:
            lst = validate_numeric_input(data.get('input', ''))
        if not lst:
            raise ValueError("List cannot be empty")
        frequency = {}
        for item in lst:
            frequency[item] = frequency.get(item, 0) + 1
        max_count = max(frequency.values())
        result = min([item for item in frequency if frequency[item] == max_count])
        return jsonify({'success': True, 'result': {'most_frequent': result}})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/exam3', methods=['POST'])
def exam3():
    """Analyzes temperature data with conversions and statistics"""
    try:
        data = request.json
        if not data:
            raise ValueError("No JSON data provided")
        if data.get('example'):
            temperatures_celsius = np.array([20, 22, 19, 25, 28, 30, 18, 21, 23, 24,
                                            26, 27, 29, 31, 17, 16, 20, 22, 25, 19,
                                            23, 24, 26, 28, 30, 15, 18, 21, 27, 29])
        else:
            input_str = data.get('input', '')
            if not input_str:
                raise ValueError("Temperature input cannot be empty")
            temperatures_celsius = np.array(validate_numeric_input(input_str))
        if len(temperatures_celsius) == 0:
            raise ValueError("Temperature array cannot be empty")
        temperatures_fahrenheit = (temperatures_celsius * 9/5) + 32
        average_temp_celsius = np.mean(temperatures_celsius)
        hottest_temp = np.max(temperatures_celsius)
        coldest_temp = np.min(temperatures_celsius)
        days_above_average = np.sum(temperatures_celsius > average_temp_celsius)
        result = {
            'average_temp_celsius': round(float(average_temp_celsius), 2),
            'hottest_temp': float(hottest_temp),
            'coldest_temp': float(coldest_temp),
            'days_above_average': int(days_above_average),
            'temperatures_fahrenheit': temperatures_fahrenheit.tolist()
        }
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/activity2', methods=['POST'])
def activity2():
    """
    Displays basic array properties:
    - Shape
    - Data type
    - Size
    """
    try:
        data = request.json
        if data.get('example'):
            array = np.array([1, 2, 3, 4, 5])  # Hardcoded: int64
        else:
            array = np.array(validate_numeric_input(data.get('input', '')))  # Now matches int64 for integer inputs
        
        result = {
            'array': array.tolist(),
            'shape': str(array[np.newaxis, :].shape),
            'dtype': str(array.dtype),
            'size': int(array.size)
        }

        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/activity3', methods=['POST'])
def activity3():
    """
    Generates a 5x5 random array and demonstrates operations:
    - Statistical calculations (mean, median, std)
    - Element-wise operations
    - Array reshaping
    No user input required - always generates a new random 5x5 array
    """
    try:
        # Generate a 5x5 array with random integers between 0 and 100
        arr = np.random.randint(0, 100, size=(5, 5))

        # Statistical calculations
        mean_val = float(np.mean(arr))
        median_val = float(np.median(arr))
        std_val = float(np.std(arr))

        # Element-wise operations with another array
        arr2 = np.ones_like(arr) * 2
        multiplication = arr * arr2

        # Reshaping
        reshaped_25x1 = arr.reshape(25, 1)

        result = {
            'original_array': arr.tolist(),
            'stats': {
                'mean': round(mean_val, 2),
                'median': round(median_val, 2),
                'std': round(std_val, 2)
            },
            'multiplied_by_2': multiplication.tolist(),
            'reshaped_25x1': reshaped_25x1.tolist()
        }
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/activity4', methods=['POST'])
def activity4():
    """
    Demonstrates array slicing and indexing operations:
    - Row and column extraction
    - Submatrix selection
    - Step slicing
    """
    try:
        data = request.json
        if data.get('example'):
            arr = np.array([[85, 90, 78, 92, 88],
                           [76, 85, 95, 88, 91],
                           [82, 79, 86, 94, 87],
                           [90, 88, 83, 91, 85]])
        else:
            arr = validate_matrix_input(data.get('input', ''))
            if arr.shape != (4, 5):
                raise ValueError("Input must be a 4x5 matrix")

        result = {
            'original_array': arr.tolist(),
            'first_row': arr[0, :].tolist(),
            'first_column': arr[:, 0].tolist(),
            'submatrix_2x2': arr[1:3, 1:3].tolist(),
            'every_other_row': arr[::2, :].tolist(),
            'high_scores': arr[arr > 90].tolist()
        }
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/activity5', methods=['POST'])
def activity5():
    """
    Normalizes an input array to values between 0 and 1 using min-max normalization
    """
    try:
        data = request.json
        if data.get('example'):
            arr = np.array([2, 5, 8, 1, 3, 9, 4, 7, 6])
        else:
            arr = np.array(validate_numeric_input(data.get('input', '')))
        
        # Normalize array
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max == arr_min:
            normalized = np.zeros_like(arr, dtype=float)
        else:
            normalized = (arr - arr_min) / (arr_max - arr_min)

        result = {
            'original_array': arr.tolist(),
            'normalized': normalized.tolist(),
            'min_value': float(normalized.min()),
            'max_value': float(normalized.max())
        }
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/activity6', methods=['POST'])
def activity6():
    """
    Performs matrix multiplication with dimension checking
    Requires two matrix inputs
    """
    try:
        data = request.json
        if data.get('example'):
            arr1 = np.array([[1, 2, 3], [4, 5, 6]])
            arr2 = np.array([[7, 8], [9, 10], [11, 12]])
        else:
            arr1 = validate_matrix_input(data.get('input1', ''))
            arr2 = validate_matrix_input(data.get('input2', ''))
        
        if arr1.shape[1] != arr2.shape[0]:
            raise ValueError(
                f"Cannot multiply matrices: incompatible dimensions "
                f"{arr1.shape} and {arr2.shape}"
            )

        result = {
            'matrix1': arr1.tolist(),
            'matrix2': arr2.tolist(),
            'product': np.matmul(arr1, arr2).tolist()
        }
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/activity7', methods=['POST'])
def activity7():
    """
    Applies mathematical operations to an array:
    - Square root
    - Natural logarithm
    - Exponential
    """
    try:
        data = request.json
        if data.get('example'):
            np.random.seed(42)
            arr = np.random.uniform(0, 100, 10)
        else:
            arr = np.array(validate_numeric_input(data.get('input', '')))
            if np.any(arr < 0):
                raise ValueError("All numbers must be non-negative")

        result = {
            'original_array': arr.tolist(),
            'sqrt': np.round(np.sqrt(arr), 3).tolist(),
            'log': np.round(np.log(arr + 1e-10), 3).tolist(),
            'exp': np.round(np.exp(arr), 3).tolist()
        }
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/activity8', methods=['POST'])
def activity8():
    """
    Calculates statistics and applies transformations:
    - Statistical measures
    - Square, sqrt, and cube transformations
    """
    try:
        data = request.json
        if data.get('example'):
            arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        else:
            arr = validate_matrix_input(data.get('input', ''))

        result = {
            'original_array': arr.tolist(),
            'stats': {
                'mean': float(np.mean(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'std': float(np.std(arr))
            },
            'transformations': {
                'square': np.square(arr).tolist(),
                'sqrt': np.sqrt(arr).tolist(),
                'cube': (arr ** 3).tolist()
            }
        }
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)