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

        public void Train(AdaData[] data, string[] featureNames)
        {
            this.featureNames = featureNames;
            Train(data);
        }

        public void Train(AdaData[] data)
        {
            if(data.Length == 0) { throw new Exception("Cannot train adaboost classifier on empty array."); }

            double[] W = new double[data.Length];
            int N = data.Length;
            int numFeatures = data[0].Length;

            for (int i = 0; i < N; i++)
            {
                W[i] = 1.0 / N;
            }


            for (int i = 0; i < NumClassifiers; i++)
            {
                double minError = double.MaxValue;
                DecisionStump bestStump = new DecisionStump(0);

                for (int j = 0; j < numFeatures; j++)
                {
                    DecisionStump testStump = new DecisionStump(j);
                    double err = testStump.Train(data, W);

                    if (err < minError)
                    {
                        minError = err;
                        bestStump = testStump;
                    }
                }

                stumps.Add(bestStump);

                int[] YHat = bestStump.Predict(data);

                double sum = 0;

                for (int j = 0; j < W.Length; j++)
                {
                    W[j] = W[j] * Math.Exp(-data[j].Label * bestStump.Alpha * YHat[j]);
                    sum += W[j];
                }

                for (int j = 0; j < W.Length; j++)
                {
                    W[j] /= sum;
                }
            }
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

        public int Predict(AdaData d)
        {
            double sum = 0.0;

            foreach(DecisionStump stump in stumps)
            {
                sum += stump.Predict(d) * stump.Alpha;
            }

            return sum > 0 ? 1 : -1;
        }

        public int[] Predict(AdaData[] data)
        {
            int[] yPred = new int[data.Length];
            
            for(int i = 0; i < yPred.Length; i++)
            {
                yPred[i] = Predict(data[i]);
            }

            return yPred;
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
