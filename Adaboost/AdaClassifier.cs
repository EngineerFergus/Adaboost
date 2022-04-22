using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Adaboost
{
    public class AdaClassifier
    {
        public int NumClassifiers { get; }

        private List<DecisionStump> stumps;
        private string[]? featureNames;

        public AdaClassifier(int numClassifiers)
        {
            NumClassifiers = numClassifiers;
            stumps = new List<DecisionStump>();
        }

        public void Train(double[][] X, int[] Y, string[] featureNames)
        {
            if(featureNames.Length != X.Length)
            {
                throw new ArgumentException($"Length of {nameof(featureNames)} and {nameof(X)} " +
                    $"must be equivalent.");
            }

            this.featureNames = featureNames;

            Train(X, Y);
        }

        public void Train(double[][] X, int[] Y)
        {
            for (int i = 0; i < X.Length; i++)
            {
                if (X[i].Length != Y.Length)
                {
                    throw new ArgumentException($"Length of each set of features in {nameof(X)} did not " +
                        $"match length of {nameof(Y)}.");
                }
            }

            double[] W = new double[Y.Length];
            int N = Y.Length;
            int numFeatures = X.Length;

            for(int i = 0; i < N; i++)
            {
                W[i] = 1.0 / N;
            }


            for(int i = 0; i < NumClassifiers; i++)
            {
                double minError = double.MaxValue;
                DecisionStump bestStump = new DecisionStump(0);

                for(int j = 0; j < numFeatures; j++)
                {
                    DecisionStump testStump = new DecisionStump(j);
                    double err = testStump.Train(X[j], W, Y);

                    if(err < minError)
                    {
                        minError = err;
                        bestStump = testStump;
                    }
                }

                stumps.Add(bestStump);

                int[] YHat = bestStump.Predict(X[bestStump.FeatureIndex]);

                double sum = 0;

                for(int j = 0; j < W.Length; j++)
                {
                    W[j] = W[j] * Math.Exp(-Y[j] * bestStump.Alpha * YHat[j]);
                    sum += W[j];
                }

                for(int j = 0; j < W.Length; j++)
                {
                    W[j] /= sum;
                }
            }
        }

        public int[] Predict(double[][] X)
        {
            int N = X[0].Length;
            int[] Y = new int[N];

            for(int i = 0; i < N; i++)
            {
                double sum = 0.0;

                foreach(DecisionStump stump in stumps)
                {
                    sum += stump.Predict(X[stump.FeatureIndex][i]) * stump.Alpha;
                }

                Y[i] = sum >= 0 ? 1 : -1;
            }

            return Y;
        }

        public int Predict(double[] X)
        {
            double sum = 0.0;

            foreach(DecisionStump stump in stumps)
            {
                sum += stump.Predict(X[stump.FeatureIndex]) * stump.Alpha;
            }

            return sum >= 0 ? 1 : -1;
        }

        public string PrintOutClassifier()
        {
            StringBuilder data = new StringBuilder();

            data.AppendLine("Adaboost classifier");
            data.AppendLine("Feature,Threshold,Parity,Alpha");

            foreach(DecisionStump stump in stumps)
            {
                if(featureNames is null)
                {
                    data.AppendLine($"{stump.FeatureIndex}, {stump.Threshold}, {stump.Parity}, {stump.Alpha}");
                }
                else
                {
                    data.AppendLine($"{featureNames[stump.FeatureIndex]} [{stump.FeatureIndex}], {stump.Threshold}, {stump.Parity}, {stump.Alpha}");
                }
            }

            return data.ToString();
        }
    }
}
