import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

def main():
    print("Loading data from eye_data.csv...")
    try:
        df = pd.read_csv('eye_data.csv')
    except FileNotFoundError:
        print("Error: eye_data.csv not found! You need to run data_collector.py first.")
        return

    # Inputs to the AI (Your eye landmark positions)
    X = df.drop(['Screen_X', 'Screen_Y'], axis=1)
    
    # What the AI should learn to predict (Screen coordinates)
    Y_x = df['Screen_X']
    Y_y = df['Screen_Y']

    print(f"Training Custom AI on {len(df)} data points...")
    print("(This uses a Random Forest Regressor, which is great at learning the 3D curves of your eyes!)")

    model_x = RandomForestRegressor(n_estimators=100, random_state=42)
    model_y = RandomForestRegressor(n_estimators=100, random_state=42)

    model_x.fit(X, Y_x)
    model_y.fit(X, Y_y)

    print("Saving your personalized AI models to disk...")
    with open('model_x.pkl', 'wb') as f:
        pickle.dump(model_x, f)
    
    with open('model_y.pkl', 'wb') as f:
        pickle.dump(model_y, f)

    print("\nSUCCESS! Your AI is trained and saved as model_x.pkl and model_y.pkl")
    print("We can now plug these into a new Smart Mouse script!")

if __name__ == "__main__":
    main()
