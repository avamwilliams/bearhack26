
from suggestions import generate_suggestions


#generate suggestions
results = generate_suggestions("normal.csv")
print(results)


#compromised payload
comp_result = generate_suggestions("compromised.csv")
print(comp_result)