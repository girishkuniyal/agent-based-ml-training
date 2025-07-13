usecase_name : iris_prediction


Goal : Predict flower species based on metric of flower. its a classification problem.
data schema:
===========
id : identity column/index
SepalLengthCm : metric	
SepalWidthCm	 : metric
PetalLengthCm : metric	
PetalWidthCm	: metric
Species : flower type



Test api:
==================
{
  "Id": 0,
  "SepalLengthCm": 5.4,
  "SepalWidthCm": 3.0,
  "PetalLengthCm": 4.5,
  "PetalWidthCm": 1.5
}

Result : Iris-versicolor