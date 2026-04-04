from inference import predict

#normal payload
result = predict("normal.csv")
print(result)


#compromised payload
comp_result = predict("compromised.csv")
print(comp_result)