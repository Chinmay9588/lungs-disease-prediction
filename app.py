from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load model
model = joblib.load('lungs.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    # Get data
    data = {
        "Age": request.form.get("Age"),
        "Gender": request.form.get("Gender"),
        "AirPollution": request.form.get("AirPollution"),
        "Alcoholuse": request.form.get("Alcoholuse"),
        "DustAllergy": request.form.get("DustAllergy"),
        "OccuPationalHazards": request.form.get("OccuPationalHazards"),
        "GeneticRisk": request.form.get("GeneticRisk"),
        "chronicLungDisease": request.form.get("chronicLungDisease"),
        "BalancedDiet": request.form.get("BalancedDiet"),
        "Obesity": request.form.get("Obesity"),
        "Smoking": request.form.get("Smoking"),
        "PassiveSmoker": request.form.get("PassiveSmoker"),
        "ChestPain": request.form.get("ChestPain"),
        "CoughingofBlood": request.form.get("CoughingofBlood"),
        "Fatigue": request.form.get("Fatigue"),
        "WeightLoss": request.form.get("WeightLoss"),
        "ShortnessofBreath": request.form.get("ShortnessofBreath"),
        "Wheezing": request.form.get("Wheezing"),
        "SwallowingDifficulty": request.form.get("SwallowingDifficulty"),
        "ClubbingofFingerNails": request.form.get("ClubbingofFingerNails"),
        "FrequentCold": request.form.get("FrequentCold"),
        "DryCough": request.form.get("DryCough"),
        "Snoring": request.form.get("Snoring")
    }

    # Convert to DataFrame
    df = pd.DataFrame([data])

    # Convert to numeric
    df = df.apply(pd.to_numeric)

    # FIX: Ensure correct column order
    df = df[[
        "Age","Gender","AirPollution","Alcoholuse","DustAllergy",
        "OccuPationalHazards","GeneticRisk","chronicLungDisease",
        "BalancedDiet","Obesity","Smoking","PassiveSmoker",
        "ChestPain","CoughingofBlood","Fatigue","WeightLoss",
        "ShortnessofBreath","Wheezing","SwallowingDifficulty",
        "ClubbingofFingerNails","FrequentCold","DryCough","Snoring"
    ]]

    # Prediction
    prediction = model.predict(df)

    # Result mapping
    if prediction[0] == 1:
        result = "🟢 LOW RISK"
    elif prediction[0] == 2:
        result = "🟡 MEDIUM RISK"
    else:
        result = "🔴 HIGH RISK"

    return render_template('index.html', prediction_text=result)


if __name__ == '__main__':
    app.run(debug=True)