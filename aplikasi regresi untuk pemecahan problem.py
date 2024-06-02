import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def linear_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_test, y_pred

def simple_power_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train_transformed = np.sqrt(X_train)  
    X_test_transformed = np.sqrt(X_test)
    model = LinearRegression()
    model.fit(X_train_transformed, y_train)
    y_pred = model.predict(X_test_transformed)
    return model, y_test, y_pred

def exponential_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    X_train_transformed = np.log(X_train.replace(0, 1e-10))  
    X_test_transformed = np.log(X_test.replace(0, 1e-10))
    
    model = LinearRegression()
    model.fit(X_train_transformed, y_train)
    y_pred = model.predict(X_test_transformed)
    return model, y_test, y_pred

def solve_problem(data_path, problem_num, method_nums):
    data = pd.read_csv(data_path)
    
    print("Loaded data:")
    print(data.head())
    
    if problem_num == 1:
        X = data[['Hours Studied']]
        problem_name = "Problem 1"
    elif problem_num == 2:
        X = data[['Sample Question Papers Practiced']]
        problem_name = "Problem 2"
    else:
        raise ValueError("Invalid problem number. Must be 1 or 2.")
    
    y = data['Performance Index']
    
    results = {}
    rmse_values = {}
    

    for method_num in method_nums:
        if method_num == 1:
            model, y_test, y_pred = linear_regression(X, y)
        elif method_num == 2:
            model, y_test, y_pred = simple_power_regression(X, y)
        elif method_num == 3:
            model, y_test, y_pred = exponential_regression(X, y)
        else:
            raise ValueError("Invalid method number. Must be 1, 2, or 3.")
        
        results[method_num] = {'model': model, 'y_test': y_test, 'y_pred': y_pred}
        
      
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_values[method_num] = rmse
    
    return results, rmse_values, problem_name

def plot_results(results):
    plt.figure(figsize=(12, 6))
    for method_num, result in results.items():
        plt.scatter(result['y_test'], result['y_pred'], label=f'Method {method_num}')
    plt.xlabel('Actual Performance Index')
    plt.ylabel('Predicted Performance Index')
    plt.title('Actual vs Predicted Performance Index')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":

    data_path = "student_performance.csv"  
    data = pd.read_csv(data_path)
    
  
    print("Loaded data:")
    print(data.head())
    
    
    nim_last_digit = int(input("Masukkan digit terakhir NIM Anda: "))
    
    if nim_last_digit % 4 == 0:
        problem_num = 1
        method_nums = [1, 3]
    elif nim_last_digit % 4 == 1:
        problem_num = 1
        method_nums = [1, 2]
    elif nim_last_digit % 4 == 2:
        problem_num = 2
        method_nums = [1, 3]
    elif nim_last_digit % 4 == 3:
        problem_num = 2
        method_nums = [1, 2]
    else:
        raise ValueError("Invalid NIM.")
    
    results, rmse_values, problem_name = solve_problem(data_path, problem_num, method_nums)
    
    print(f"NIM Anda ({nim_last_digit}) mengerjakan {problem_name}")
    
    plot_results(results)
    
    for method_num, rmse in rmse_values.items():
        print(f"Metode {method_num} RMSE: {rmse}")
