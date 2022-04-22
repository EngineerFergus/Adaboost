// See https://aka.ms/new-console-template for more information

using Adaboost;

Console.WriteLine("Hello, World!");
CSVData data = CSVRead.ReadCSV("C:\\temp\\ionosphere.csv");
AdaClassifier classifier = new AdaClassifier(50);

for(int i = 0; i < data.Labels.Length; i++)
{
    if(data.Labels[i] == 0) { data.Labels[i] = -1; }
}

classifier.Train(data.Features, data.Labels, data.Headers);
int[] predictions = classifier.Predict(data.Features);

int correct = 0;

for(int i = 0; i < predictions.Length; i++)
{
    if(predictions[i] == data.Labels[i]) { correct++; }
}

Console.WriteLine($"Classifier Accuracy: {correct} / {data.Labels.Length}");
Console.WriteLine(classifier.PrintOutClassifier());
