using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Adaboost
{
    public class AdaDataSet
    {
        public Dictionary<string, int> Labels;
        public string[] FeatureNames;
        public AdaData[] Data;

        public AdaDataSet(Dictionary<string, int> labels, string[] featureNames, AdaData[] data)
        {
            Labels = labels;
            FeatureNames = featureNames;
            Data = data;
        }
    }
}
