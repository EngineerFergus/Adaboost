using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Adaboost
{
    public class LabeledData
    {
        private double[] features;
        public int Label { get; }

        public double this[int index]
        {
            get
            {
                return features[index];
            }
        }

        public LabeledData(double[] features, int label)
        {
            Label = label;
            this.features = features;
        }

        public int[] GetOneHotLabel(int numClasses)
        {
            if(numClasses <= Label)
            {
                throw new ArgumentException($"Exception in {nameof(GetOneHotLabel)}: Cannot convert to " +
                    $"one hot encoding when label is greater than total number of classes.");
            }

            if(Label < 0)
            {
                throw new Exception($"Exception in {nameof(GetOneHotLabel)}: Cannot convert to one hot " +
                    $"encoding with a label less than zero.");
            }

            int[] oneHots = new int[numClasses];
            oneHots[Label] = 1;
            return oneHots;
        }
    }
}
