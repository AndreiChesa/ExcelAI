import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt

class DatasetTraining:
    def __init__(self, file_path):
        self.house_data = pd.read_csv(file_path)

    def question_test(self):
        return self.house_data.describe()

    def question_a(self):
        return len(self.house_data)
    
    def question_b(self):
        return self.house_data.columns.tolist()
    
    def question_c(self):
        return [col for col in self.house_data.columns if self.house_data[col].dtype == int][:2]

    def question_d(self):
        return self.house_data.select_dtypes(include=['object']).columns.tolist()[:3]

    def question_e(self):
        return self.house_data['Age'].mean()
    
    def question_f(self):
        return self.house_data['Fare'].max()
    
    def question_g(self):
        return self.house_data['Survived'].value_counts()[1]

    def question_h(self):
        return self.house_data['Age'].max()
    
    def question_i(self):
        return self.house_data['Age'].min()
    
    def question_j(self):
        return self.house_data['Age'].isna().sum()

    def question_k(self):
        return self.house_data['Embarked'].isna().sum()
    
    def question_l(self):
        return 'male' if self.house_data['Sex'].mode()[0] == 'male' else 'female'
    
    def question_m(self):
        return len(self.house_data[(self.house_data['Survived'] == 1) & (self.house_data['Age'] < 18)])

    def question_n(self):
        return self.house_data['Fare'].sum()
    
    def question_two_a(self):
        plt.hist(self.house_data['Age'].dropna(), bins=30, edgecolor='green')
        plt.title('Age Distribution')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.show()

    def question_two_b(self):
        plt.hist(self.house_data['Fare'].dropna(), bins=30, edgecolor='red')
        plt.title('Fare Distribution')
        plt.xlabel('Fare')
        plt.ylabel('Frequency')
        plt.show()

    def question_two_c(self):
        survival_counts = self.house_data['Survived'].value_counts()
        plt.pie(survival_counts.values, labels=['Not Survived', 'Survived'], autopct='%1.1f%%')
        plt.title('Survival Distribution')
        plt.show()

    def question_two_d(self):
        male_data = self.house_data[self.house_data['Sex'] == 'male']
        male_survival_counts = male_data['Survived'].value_counts()
        plt.pie(male_survival_counts.values, labels=['Not Survived', 'Survived'], autopct='%1.1f%%')
        plt.title('Male Survival Distribution')
        plt.show()

    def question_two_e(self):
        female_data = self.house_data[self.house_data['Sex'] == 'female']
        female_survival_counts = female_data['Survived'].value_counts()
        plt.pie(female_survival_counts.values, labels=['Not Survived', 'Survived'], autopct='%1.1f%%')
        plt.title('Female Survival Distribution')
        plt.show()

    def question_two_f(self):
        data = self.house_data['Embarked'].dropna().value_counts()
        plt.bar(data.index, data.values, color=['skyblue', 'lightcoral', 'lightgreen'])
        plt.title('Distribution of Embarked Ports')
        plt.xlabel('Embarked Port (C=Cherbourg, Q=Queenstown, S=Southampton)')
        plt.ylabel('Number of Passengers')
        plt.show()


test = DatasetTraining('H6/titanic.csv')

def part_one():
    print("Question Test (Data Description):", test.question_test())
    print("Question A (Number of rows):", test.question_a())
    print("Question B (Column names):", test.question_b())
    print("Question C (Integer columns):", test.question_c())
    print("Question D (Object columns):", test.question_d())
    print("Question E (Mean of 'Age'):", test.question_e())
    print("Question F (Max of 'Fare'):", test.question_f())
    print("Question G (Survived counts):", test.question_g())
    print("Question H (Max of 'Age'):", test.question_h())
    print("Question I (Min of 'Age'):", test.question_i())
    print("Question J (Count of NaN in 'Age'):", test.question_j())
    print("Question K (Count of NaN in 'Embarked'):", test.question_k())
    print("Question L (Most common 'Sex'):", test.question_l())
    print("Question M (Count of child survivors):", test.question_m())
    print("Question N (Sum of 'Fare'):", test.question_n())

def part_two():
    test.question_two_a()
    test.question_two_b()
    test.question_two_c()
    test.question_two_d()
    test.question_two_e()
    test.question_two_f()


part_two()