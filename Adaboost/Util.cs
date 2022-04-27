using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Adaboost
{
    public class Util
    {
        public static AdaDataSet ReadAdaData(string dir)
        {
            List<double[]> data = new List<double[]>();
            List<string> labels = new List<string>();
            List<string> featureNames = new List<string>();

            using (FileStream stream = File.OpenRead(dir))
            {
                using (StreamReader reader = new StreamReader(stream))
                {
                    int count = 0;

                    while (!reader.EndOfStream)
                    {
                        string? line = reader.ReadLine();
                        if (line == null) { continue; }
                        string[] splits = line.Split(',');
                        int numCols = splits.Length;

                        if (count == 0)
                        {
                            for (int i = 0; i < numCols - 1; i++)
                            {
                                featureNames.Add(splits[i]);
                            }
                        }
                        else
                        {
                            double[] features = new double[numCols - 1];

                            for (int i = 0; i < splits.Length - 1; i++)
                            {
                                _ = double.TryParse(splits[i], out double f);
                                features[i] = f;
                            }

                            data.Add(features);
                            labels.Add(splits.Last());
                        }

                        count++;
                    }
                }
            }

            List<string> distinctLabels = labels.Distinct().ToList();

            if(distinctLabels.Count != 2) {
                throw new Exception($"Exception in {nameof(ReadAdaData)}, cannot" +
                    $"load more than 2 classes for adaboost datasets.");
            }

            Dictionary<string, int> labelDict = new Dictionary<string, int>();

            labelDict.Add(distinctLabels[0], -1);
            labelDict.Add(distinctLabels[1], 1);

            AdaData[] adaDatas = new AdaData[data.Count];

            for(int i = 0; i < adaDatas.Length; i++)
            {
                adaDatas[i] = new AdaData(data[i], labelDict[labels[i]]);
            }
            
            return new AdaDataSet(labelDict, featureNames.ToArray(), adaDatas);
        }
    }
}
