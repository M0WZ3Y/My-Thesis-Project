try:
    import pandas
    print("SUCCESS: Pandas is installed!")
    import numpy
    print("SUCCESS: NumPy is installed!")
    import matplotlib
    print("SUCCESS: Matplotlib is installed!")
    import sklearn
    print("SUCCESS: Scikit-learn is installed!")
    print("\nAll dependencies are ready!")
except ImportError as e:
    print(f"Still missing: {e}")
    print("Installation still in progress...")