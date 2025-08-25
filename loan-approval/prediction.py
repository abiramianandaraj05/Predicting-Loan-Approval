import prediction_notebook
import pandas as pd

def setUp():
    model = prediction_notebook.getModel()
    return model

def createData(gender, marriage,education,applicant_income,coapplicant_income,loan_amount,loan_amount_term,credit_history,property_area ):
    data = {'Gender':[gender],'Married':[marriage],'Education':[education],'ApplicantIncome':[applicant_income],'CoapplicantIncome':[coapplicant_income],'LoanAmount':[loan_amount],'Loan_Amount_Term':[loan_amount_term],'Credit_History':[credit_history],'Property_Area':[property_area]}
    return data

def convert_to_dataframe(dataset):
    return pd.DataFrame(data = dataset)

def getGender():
    text = "Type\nM: Male\nF:Female "
    userAnwser = validateInput(text,'M','F')
    if userAnwser == "M":
        return 1.0
    else:
        return 0.0
    
def getMarrigeData():
    text = "Type\nY:Yes I am Married\nN:No I am not married "
    userAnwser = validateInput(text,'Y','N')
    if userAnwser == "Y":
        return 1.0
    else:
        return 0.0
    
def getEducation():
    text = "Type\nY:Yes I am a Graduate\nN:No I am not a Graduate "
    userAnwser = validateInput(text,'Y','N')
    if userAnwser == "N":
        return 1.0
    else:
        return 0.0
    
def getNumber(text):
    valid = False
    while not valid:
        try:
            userAnwser = float(input(text))
            if userAnwser >= 0:
                valid = True
        except:
            pass
        
    return userAnwser 

def getIncome():
    text = "Enter your Income: "
    return getNumber(text)

def getCo_Applicant_Income():
    text = "Enter your Co Applicant Income: "
    return getNumber(text)

def getLoanAmount():
    text = "Enter your Loan Amount: "
    return getNumber(text)

def get_Loan_Amount_Term():
    text = "Enter your Loan Amount Term: "
    return getNumber(text)

def get_Credit_History():
    text = "Enter your Credit History: "
    return getNumber(text)

def get_Property_Area():
    userAnwser = input("What property Area do you live in? Type\nU:Urban\nR:Rural\nS:Semiurban ").upper()
    while userAnwser != 'S'and userAnwser != 'U' and userAnwser != 'R':
        print("Invalid Input try again\n")
        userAnwser = input("What property Area do you live in? Type\nU:Urban\nR:Rural\nS:Semiurban ").upper()
    if userAnwser == "U":
        return 2
    elif userAnwser == "R":
        return 0
    else:
        return 1


    
def validateInput(text,char1,char2):
    userAnwser = input(text).upper()
    while userAnwser != char1 and userAnwser != char2:
        print("Invalid Anwser\n")
        userAnwser = input(text).upper()
    return userAnwser
    

def get_User_Data():
    return getGender(), getMarrigeData(), getEducation(),getIncome(),getCo_Applicant_Income(),getLoanAmount(),get_Loan_Amount_Term(),get_Credit_History(),get_Property_Area()

def Main():
    model = setUp()
    gender, marriage,education,applicant_income,coapplicant_income,loan_amount,loan_amount_term,credit_history,property_area = get_User_Data()
    data = createData(gender, marriage,education,applicant_income,coapplicant_income,loan_amount,loan_amount_term,credit_history,property_area)
    df = convert_to_dataframe(data)
    print(df)
    prediction = model.predict(df)
    if int(prediction) == 1:
        print("You may be approved for a loan \n please submit an application to further check if you can")
    else:
        print("You will not be approved for a loan") 
    return


Main()