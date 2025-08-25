import prediction_notebook
import pandas as pd

# this code gets the previous trained model from the notebook
def setUp():
    model = prediction_notebook.getModel()
    return model

# this puts the data inputted by the users into a set with each feature and its corresponding value
def createData(gender, marriage,education,applicant_income,coapplicant_income,loan_amount,loan_amount_term,credit_history,property_area ):
    data = {'Gender':[gender],'Married':[marriage],'Education':[education],'ApplicantIncome':[applicant_income],'CoapplicantIncome':[coapplicant_income],'LoanAmount':[loan_amount],'Loan_Amount_Term':[loan_amount_term],'Credit_History':[credit_history],'Property_Area':[property_area]}
    return data

#this code converts the data into a dataframe so it can be inputted into the model
def convert_to_dataframe(dataset):
    return pd.DataFrame(data = dataset)

# gets the gender and if male returns 1 and returns 0 for female
def getGender():
    text = "Type\nM: Male\nF:Female "
    userAnwser = validateInput(text,'M','F')
    if userAnwser == "M":
        return 1.0
    else:
        return 0.0

# gets whether a user is married, Y returns 1 and N returns 0
def getMarrigeData():
    text = "Type\nY:Yes I am Married\nN:No I am not married "
    userAnwser = validateInput(text,'Y','N')
    if userAnwser == "Y":
        return 1.0
    else:
        return 0.0
    
# gets whether a user graduated or not returns 0 if person hasnt graduated and 1 if they have
def getEducation():
    text = "Type\nY:Yes I am a Graduate\nN:No I am not a Graduate "
    userAnwser = validateInput(text,'Y','N')
    if userAnwser == "N":
        return 1.0
    else:
        return 0.0

# gets a number from the user any number less than 0 it will repeatedly ask the user for a new number 
# until a number greater than equal to 0 is entered 
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

# gets the user income and return the number
def getIncome():
    text = "Enter your Income: "
    return getNumber(text)

#gets the co applicants income and returns the number
def getCo_Applicant_Income():
    text = "Enter your Co Applicant Income: "
    return getNumber(text)

# gets the loan amount from the user and returns the number
def getLoanAmount():
    text = "Enter your Loan Amount: "
    return getNumber(text)

# gets the loan amount term and returns the number
def get_Loan_Amount_Term():
    text = "Enter your Loan Amount Term: "
    return getNumber(text)

# gets the credit history and returns the number
def get_Credit_History():
    text = "Enter your Credit History: "
    return getNumber(text)

# asks whether the user lives in an urban rural or semiurban area
# returns 0 for rural and 2 for urban and 1 for semiurban
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

# this function ouputs an input message to the user then checks whether the input is appropriate
# if it is innapropriate input then the input is repeated until a valid input is entered    
def validateInput(text,char1,char2):
    userAnwser = input(text).upper()
    while userAnwser != char1 and userAnwser != char2:
        print("Invalid Anwser\n")
        userAnwser = input(text).upper()
    return userAnwser
    

# gets all the data from the user
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