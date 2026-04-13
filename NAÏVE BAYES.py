import pandas as pd
data = {
'Offer': ['Yes','Yes','No','No','Yes','No','Yes','No'],
'Money': ['Yes','No','Yes','No','Yes','No','Yes','Yes'],
'Spam': ['Yes','Yes','No','No','Yes','No','No','Yes']
}
df = pd.DataFrame(data)
print("Dataset:\n", df)
# New email classify
test_offer = 'Yes'
test_money = 'Yes'
# Prior probability
total = len(df)
spam_count = len(df[df['Spam']=='Yes'])
notspam_count = len(df[df['Spam']=='No'])
P_spam = spam_count/total
P_notspam = notspam_count/total
print("\n Prior Probabilities")
print("P(Spam) =", P_spam)
print("P(NotSpam) =", P_notspam)
# Likelihood probability
spam_df = df[df['Spam']=='Yes']
notspam_df = df[df['Spam']=='No']
P_offer_spam = len(spam_df[spam_df['Offer']==test_offer]) / spam_count
P_money_spam = len(spam_df[spam_df['Money']==test_money]) / spam_count
P_offer_notspam = len(notspam_df[notspam_df['Offer']==test_offer]) / notspam_count
P_money_notspam = len(notspam_df[notspam_df['Money']==test_money]) / notspam_count
print("\n Likelihood Probabilities")
print("P(Offer=Yes | Spam) =", P_offer_spam)
print("P(Money=Yes | Spam) =", P_money_spam)
print("P(Offer=Yes | NotSpam) =", P_offer_notspam)
print("P(Money=Yes | NotSpam) =", P_money_notspam)
# Naive Bayes Classifier Formula
P_spam_X = P_spam * P_offer_spam * P_money_spam
P_notspam_X = P_notspam * P_offer_notspam * P_money_notspam
print("\n Posterior Probabilities")
print("P(Spam | X) =", P_spam_X)
print("P(NotSpam | X) =", P_notspam_X)
# Final Prediction
print("\n Final Prediction:")
if P_spam_X > P_notspam_X:
print("Email is Spam")
else:
print("Email is Not Spam")
