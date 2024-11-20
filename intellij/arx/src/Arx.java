import datasets.ACSIncome;
import datasets.Dataset;
import datasets.Patients;
import org.deidentifier.arx.*;
import org.deidentifier.arx.aggregates.StatisticsEquivalenceClasses;
import org.deidentifier.arx.aggregates.StatisticsQuality;
import org.deidentifier.arx.criteria.KAnonymity;
import org.deidentifier.arx.criteria.PrivacyCriterion;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;
import java.util.stream.Collectors;


public class Arx {

    public static void printResults(ARXResult result, String outPath, boolean filterSuppressed, String statisticsOutPath, boolean calculateStatistics) throws IOException {
        File folder = new File(outPath);
        folder.mkdirs();
        boolean printedFullSuppressed = false;

        if (calculateStatistics){
            File folder2 = new File(statisticsOutPath);
            folder2.mkdirs();
        }

        ARXLattice lattice = result.getLattice();
        List<ARXLattice.ARXNode> nodes = new ArrayList<>();
        for (ARXLattice.ARXNode[] arxNodes : lattice.getLevels()) {
            for (ARXLattice.ARXNode arxNode : arxNodes) {
                lattice.expand(arxNode);
                nodes.add(arxNode);
            }
        }
        // print applied privacy models, qid, sa
        printSettings(result, outPath);

        for(ARXLattice.ARXNode node:nodes){
            if(!filterSuppressed) {
                // prints dataset
                printResult(result, node, outPath);
                if(calculateStatistics) {
                    printStatistics(result, node, statisticsOutPath);
                }
            }else {
                if (!node.isChecked()) {
                    result.getOutput(node);
                }
                if (!node.getLowestScore().toString().equals("1.0")) {
                    // if not full suppressed print
                    printResult(result, node, outPath);
                    if(calculateStatistics) {
                        printStatistics(result, node, statisticsOutPath);
                    }
                }else if(!printedFullSuppressed){
                    // print a single full suppressed dataset
                    printedFullSuppressed = true;
                    printResult(result, node, outPath);
                    if(calculateStatistics) {
                        printStatistics(result, node, statisticsOutPath);
                    }
                }
            }
        }
    }

    public static void printResult(ARXResult result, ARXLattice.ARXNode node, String outpath) throws IOException {
        if (node.getAnonymity().equals(ARXLattice.Anonymity.ANONYMOUS)) {
            DataHandle output = result.getOutput(node);
            // constuct a name for the file based on the generalization node
            String nodeName = Arrays.stream(node.getTransformation()).mapToObj(String::valueOf).collect(Collectors.joining("_"));

            output.save(outpath+nodeName+".csv",',');
        }
    }

    public static void printStatistics(ARXResult result, ARXLattice.ARXNode node, String printStatistics){
        // print the following as a csv
        // k, #suppressed, avg class size, #eqs, total gen level, precision, granularity, non-uniform entropy, squared error, discernibility, ambiguity, record-level squared error
        if (node.getAnonymity().equals(ARXLattice.Anonymity.ANONYMOUS)) {
            // constuct a name for the file based on the generalization node
            String nodeName = Arrays.stream(node.getTransformation()).mapToObj(String::valueOf).collect(Collectors.joining("_"));
            String outPath = printStatistics+nodeName+".csv";

            String header = "k,#suppressed,avg class size,min class size,max class size,#eqs,total gen level,precision,granularity/loss,non-uniform entropy,squared error,discernibility,ambiguity,record-level squared error";

            DataHandle output = result.getOutput(node);
            List<String> values = new ArrayList<>();

            values.add(String.valueOf(((KAnonymity)result.getConfiguration().getPrivacyModels().iterator().next()).getK()));

            StatisticsEquivalenceClasses eqStats = output.getStatistics().getEquivalenceClassStatistics();
            String doubleFormat = "%.10f";

            values.add(String.valueOf(eqStats.getNumberOfSuppressedRecords()));
            values.add(String.format(doubleFormat, eqStats.getAverageEquivalenceClassSize()));
            values.add(String.valueOf(eqStats.getMinimalEquivalenceClassSize()));
            values.add(String.valueOf(eqStats.getMaximalEquivalenceClassSize()));
            values.add(String.valueOf(eqStats.getNumberOfEquivalenceClasses()));
            values.add(String.valueOf(node.getTotalGeneralizationLevel()));

            StatisticsQuality qualityStats = output.getStatistics().getQualityStatistics();
            values.add(String.format(doubleFormat, qualityStats.getGeneralizationIntensity().getArithmeticMean()));
            values.add(String.format(doubleFormat, qualityStats.getGranularity().getArithmeticMean()));
            values.add(String.format(doubleFormat, qualityStats.getNonUniformEntropy().getArithmeticMean()));
            values.add(String.format(doubleFormat, qualityStats.getAttributeLevelSquaredError().getArithmeticMean()));
            values.add(String.format(doubleFormat, qualityStats.getDiscernibility().getValue()));
            values.add(String.format(doubleFormat, qualityStats.getAmbiguity().getValue()));
            values.add(String.format(doubleFormat, qualityStats.getRecordLevelSquaredError().getValue()));

            String valuesS = String.join(",",values);

            try(PrintWriter pw = new PrintWriter(outPath)){
                pw.println(header);
                pw.println(valuesS);
            } catch (Exception e){
                e.printStackTrace();
            }

        }
    }


    private static void writeConfigs(List<Map<String, Double>> configs, String outpath) {
        File folder = new File(outpath);
        folder.mkdirs();

        try(PrintWriter pw = new PrintWriter(outpath+"configs.csv")){
            for(int i=0;i<configs.size();i++){
                pw.println(i + ";" + configs.get(i));
            }
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    public static List<String> getOrdered(DataHandle handle, Collection<String> attributes) {
        List<String> result = new ArrayList<>(attributes);
        result.sort(Comparator.comparingInt(handle::getColumnIndexOf));
        return result;
    }

    public static void printSettings(ARXResult result, String outPath){
        String header = "privacy-models;qid;sa";
        String values = "";

        Set <PrivacyCriterion> criterion = result.getConfiguration().getPrivacyModels();
        values += criterion.stream().map(PrivacyCriterion::toString).collect(Collectors.toList()).toString();
        List<String> qid = getOrdered(result.getInput(),result.getDataDefinition().getQuasiIdentifyingAttributes());
        List<String> sa = getOrdered(result.getInput(),result.getDataDefinition().getSensitiveAttributes());

        values += ";" + qid;
        values += ";" + sa;

        try(PrintWriter pw = new PrintWriter(outPath+"settings.csv")){
            pw.println(header);
            pw.println(values);
        }catch (Exception e){
            e.printStackTrace();
        }

    }

    public static void generateCombinations(List<double[]> paramLists, List<double[]> result, int depth, double[] current) {
        if (depth == paramLists.size()) {
            result.add(current.clone());
            return;
        }

        for (int i = 0; i < paramLists.get(depth).length; i++) {
            current[depth] = paramLists.get(depth)[i];
            generateCombinations(paramLists, result, depth + 1, current);
        }
    }


    public static void main(String[] args) throws Exception {

        String inputPath = "../output/ACSIncome/datasets/input/train.csv";
        String testName = "ACSIncome";
        String outPath = "../output/ACSIncome/datasets/Anonymized/";
        String printAllResultsS = "False";
        String statisticsOutPath = "../output/ACSIncome/arx_stats/Anonymized/";
        boolean printSpecificNode = false;
        int[] transformation = {};

        //inputPath = "../Datasets/ACSIncome.csv";
        //testName = "ACSIncome";
        //outPath = "../AnonymizedDatasets/ACSIncome/";


        if(args.length==5) {
            inputPath = args[0];
            testName = args[1];
            outPath = args[2];
            printAllResultsS = args[3];
            statisticsOutPath = args[4];
        }else if(args.length==4) {
            inputPath = args[0];
            testName = args[1];
            outPath = args[2];
            printAllResultsS = args[3];
            statisticsOutPath = "";
        }else if(args.length!=0){
            System.out.println("4 or 5 Arguments expected (inputPath, testName[patients|ACSIncome], outPath, printAllResults, (statisticsOutPath)) got " + args.length);
        }else{
            System.out.println("Running default settings: " + inputPath + " " + testName + " " + outPath + " " + printAllResultsS + " " + statisticsOutPath);
            //printSpecificNode = true;
            // all max generalized
            //transformation = new int[]{5,2,4,2,4,2,1,2};
        }

        Dataset dataset = null;
        if(testName.equals("patients")){
            dataset = new Patients(inputPath, outPath, statisticsOutPath);
        }
        if(testName.equals("ACSIncome")){
            dataset = new ACSIncome(inputPath, outPath, statisticsOutPath);
        }

        if(dataset==null){
            System.out.println("Invalid testName. Must be 'patients' or 'ACSIncome'.");
            return;
        }

        //Dataset dataset = new ACSIncome();
        boolean printAllResults = Boolean.parseBoolean(printAllResultsS);
        // when true, only the non generalized full suppressed will be printed all other full suppressed will not be printed
        boolean filterSuppressedData = true;

        Map<String,double[]> configParams = dataset.getParams();
        List<Map.Entry<String, double[]>> params = new ArrayList<>(configParams.entrySet());
        List<double[]> paramValues = params.stream().map(Map.Entry::getValue).collect(Collectors.toList());
        List<String> paramNames = params.stream().map(Map.Entry::getKey).collect(Collectors.toList());

        // needed when more params are required such as for l-div to create all combinations of these params
        List<double[]> combinations = new ArrayList<>();
        generateCombinations(paramValues, combinations,0,new double[configParams.size()]);

        List<Map<String,Double>> configs = new ArrayList<>();

        for(double[] combination:combinations){
            Map<String,Double> config = new HashMap<>();
            for(int i=0;i<params.size();i++){
                config.put(paramNames.get(i),combination[i]);
            }
            configs.add(config);
        }

        writeConfigs(configs,dataset.getOutPath());

        for(int i=0;i<configs.size();i++){
            Map<String,Double> config = configs.get(i);

            String mapAsString = config.keySet().stream()
                    .map(key -> key + "=" + config.get(key))
                    .collect(Collectors.joining(", ", "{", "}"));
            System.out.println("Running config: " + mapAsString);

            ARXAnonymizer anonymizer = new ARXAnonymizer();
            Data d = dataset.getData(false);
            ARXConfiguration conf = dataset.getConfig(config);
            ARXResult result = anonymizer.anonymize(d,conf);

            String fullOutPath = dataset.getOutPath() + i + "/";
            String fullStatisticsOutPath = dataset.getStatisticsOutPath() + i + "/";
            boolean calculateStatistics = !dataset.getStatisticsOutPath().isEmpty();
            if (printAllResults){
                printResults(result, fullOutPath, filterSuppressedData, fullStatisticsOutPath, calculateStatistics);
            }else {
                ARXLattice.ARXNode node;
                if (printSpecificNode){
                    node = result.getLattice().getNode(transformation);
                }else {
                    node = result.getGlobalOptimum();
                }
                System.out.println(Arrays.toString(node.getTransformation()));
                File folder = new File(fullOutPath);
                folder.mkdirs();
                printSettings(result, fullOutPath);
                printResult(result, node, fullOutPath);
                if (calculateStatistics) {
                    File folder2 = new File(fullStatisticsOutPath);
                    folder2.mkdirs();
                    printStatistics(result, node, fullStatisticsOutPath);
                }
            }
        }

    }

}
