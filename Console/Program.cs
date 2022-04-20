// See https://aka.ms/new-console-template for more information

using Adaboost;

Console.WriteLine("Hello, World!");

AdaClassifier booster = new AdaClassifier(5);

double[][] X = new double[5][];

for(int i = 0; i < 5; i++)
{
    X[i] = new double[20];
}

int[] Y = new int[20];

booster.Train(X, Y);

