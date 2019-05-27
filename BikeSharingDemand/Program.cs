using BikeSharingDemand.Helpers;
using BikeSharingDemand.ModelNamespace;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using System;
using System.Collections.Generic;
using System.Linq;

namespace BikeSharingDemand
{
    class Program
    {
        private static MLContext _mlContext = new MLContext();
        private static Dictionary<Model, double> _stats = new Dictionary<Model, double>();
        private static string _trainingDataLocation = @"Data/hour_train.csv";
        private static string _testDataLocation = @"Data/hour_test.csv";

        static void Main(string[] args)
        {

            var regressors = new List<IEstimator<ITransformer>>()
            {
                _mlContext.Regression.Trainers.Sdca(labelColumnName: "Count", featureColumnName: "Features"),
                _mlContext.Regression.Trainers.LbfgsPoissonRegression(labelColumnName: "Count", featureColumnName: "Features"),
                _mlContext.Regression.Trainers.FastForest(labelColumnName: "Count", featureColumnName: "Features"),
                _mlContext.Regression.Trainers.FastTree(labelColumnName: "Count", featureColumnName: "Features"),
                _mlContext.Regression.Trainers.FastTreeTweedie(labelColumnName: "Count", featureColumnName: "Features"),
                _mlContext.Regression.Trainers.Gam(labelColumnName: "Count", featureColumnName: "Features")
            };

            regressors.ForEach(RunAlgorythm);

            var bestModel = _stats.Where(x => x.Value == _stats.Max(y => y.Value)).Single().Key;
            VisualizeTenPredictionsForTheModel(bestModel);
            bestModel.SaveModel();

            Console.ReadLine();
        }

        private static void RunAlgorythm(IEstimator<ITransformer> algorythm)
        {
            var model = new Model(_mlContext, algorythm, _trainingDataLocation);
            model.BuildAndFit();
            PrintAndStoreMetrics(model);
        }

        private static void PrintAndStoreMetrics(Model model)
        {
            var metrics = model.Evaluate(_testDataLocation);

            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Metrics for {model.Name}          ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       R2 Score: {metrics.RSquared:#.##}");
            Console.WriteLine($"*       Mean Absolute Error: {metrics.MeanAbsoluteError:#.##}");
            Console.WriteLine($"*       Mean Squared Error: {metrics.MeanSquaredError:#.##}");
            Console.WriteLine($"*       RMS Error: {metrics.RootMeanSquaredError:#.##}");
            Console.WriteLine($"*************************************************");

            _stats.Add(model, metrics.RSquared);
        }

        private static void VisualizeTenPredictionsForTheModel(Model model)
        {
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"* BEST MODEL IS: {model.Name}!");
            Console.WriteLine($"* Here are its predictions: ");
            var testData = new BikeSharingDemandsCsvReader().GetDataFromCsv(_testDataLocation).ToList();
            for (int i = 0; i < 10; i++)
            {
                var prediction = model.Predict(testData[i]);
                Console.WriteLine($"*------------------------------------------------");
                Console.WriteLine($"* Predicted : {prediction.Score}");
                Console.WriteLine($"* Actual:    {testData[i].Count}");
                Console.WriteLine($"*------------------------------------------------");
            }
            Console.WriteLine($"*************************************************");
        }
    }
}
