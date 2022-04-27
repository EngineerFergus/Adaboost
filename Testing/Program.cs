// See https://aka.ms/new-console-template for more information

using Adaboost;

Console.WriteLine("Hello, World!");
AdaDataSet dataSet = Util.ReadAdaData("C:\\temp\\ionosphere.csv");
AdaClassifier classifier = new AdaClassifier(50);

classifier.Train(dataSet.Data, dataSet.FeatureNames);
int[] predictions = classifier.Predict(dataSet.Data);

int correct = 0;

for(int i = 0; i < predictions.Length; i++)
{
    if(predictions[i] == dataSet.Data[i].Label) { correct++; }
}

Console.WriteLine($"Classifier Accuracy: {correct} / {dataSet.Data.Length}");
Console.WriteLine(classifier.PrintOutClassifier());
