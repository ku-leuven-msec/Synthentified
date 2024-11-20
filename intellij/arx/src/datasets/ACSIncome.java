package datasets;

import org.deidentifier.arx.*;
import org.deidentifier.arx.aggregates.AggregateFunction;
import org.deidentifier.arx.aggregates.HierarchyBuilderIntervalBased;
import org.deidentifier.arx.criteria.KAnonymity;
import org.deidentifier.arx.metric.Metric;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.stream.IntStream;

public class ACSIncome extends Dataset {

    public ACSIncome(String input, String output, String statisticsOutput){
        this.outPath = output;
        this.inPath = input;
        this.statisticsOutPath = statisticsOutput;
    }

    public ARXConfiguration getConfig(Map<String,Double> configParams){
        ARXConfiguration config = ARXConfiguration.create();
        config.setQualityModel(Metric.createLossMetric(0.5, Metric.AggregateFunction.ARITHMETIC_MEAN));

        config.addPrivacyModel(new KAnonymity(configParams.get("k").intValue()));
        config.setSuppressionLimit(1);

        return config;
    }

    public Data getData(boolean generateHierarchies) throws IOException {
        DataSource source = DataSource.createCSVSource(this.inPath, StandardCharsets.UTF_8, ',', true);
        source.addColumn("AGEP", DataType.INTEGER, true);
        source.addColumn("COW", DataType.STRING, true);
        source.addColumn("SCHL", DataType.STRING, true);
        source.addColumn("MAR", DataType.STRING, true);
        source.addColumn("OCCP", DataType.STRING, true);
        source.addColumn("POBP", DataType.STRING, true);
        source.addColumn("WKHP", DataType.INTEGER, true);
        source.addColumn("SEX", DataType.STRING, true);
        source.addColumn("RAC1P", DataType.STRING, true);
        source.addColumn("PINCP", DataType.INTEGER, true);

        Data data = Data.create(source);

        data.getDefinition().setAttributeType("AGEP", AttributeType.QUASI_IDENTIFYING_ATTRIBUTE);
        data.getDefinition().setAttributeType("COW", AttributeType.QUASI_IDENTIFYING_ATTRIBUTE);
        data.getDefinition().setAttributeType("SCHL", AttributeType.QUASI_IDENTIFYING_ATTRIBUTE);
        data.getDefinition().setAttributeType("MAR", AttributeType.QUASI_IDENTIFYING_ATTRIBUTE);
        data.getDefinition().setAttributeType("OCCP", AttributeType.QUASI_IDENTIFYING_ATTRIBUTE);
        data.getDefinition().setAttributeType("POBP", AttributeType.QUASI_IDENTIFYING_ATTRIBUTE);
        data.getDefinition().setAttributeType("SEX", AttributeType.QUASI_IDENTIFYING_ATTRIBUTE);
        data.getDefinition().setAttributeType("RAC1P", AttributeType.QUASI_IDENTIFYING_ATTRIBUTE);

        data.getDefinition().setAttributeType("WKHP", AttributeType.INSENSITIVE_ATTRIBUTE);

        // are sensitive but I am not applying any privacy model on them (must be set on insensitive or we get errors)
        data.getDefinition().setAttributeType("PINCP", AttributeType.INSENSITIVE_ATTRIBUTE);


        if(generateHierarchies) {
            // the more complex and categorical hierarchies will just be read from the file instead of build in code
            String hierarchyPath = "../Hierarchies/ACSIncome/";
            File dir = new File(hierarchyPath);
            dir.mkdirs();

            // AGEP
            {
                HierarchyBuilderIntervalBased<Long> builderIntervalBased = HierarchyBuilderIntervalBased.create(DataType.INTEGER, new HierarchyBuilderIntervalBased.Range<>(1L, 1L, 1L),
                        new HierarchyBuilderIntervalBased.Range<>(101L, 101L, 101L));

                builderIntervalBased.setAggregateFunction(AggregateFunction.forType(DataType.INTEGER).createIntervalFunction(true, false));

                builderIntervalBased.addInterval(1L, 6L);


                for (int i = 0; i < 3; i++) {
                    builderIntervalBased.getLevel(i).addGroup(2);
                }

                String[] tmp = IntStream.range(1, 101).mapToObj(String::valueOf).toArray(String[]::new);

                AttributeType.Hierarchy hierarchy = builderIntervalBased.build(tmp);
                data.getDefinition().setHierarchy("AGEP", hierarchy);
                hierarchy.save(hierarchyPath + "AGEP.csv");
            }

            // read all others from the files
            String[] others = {"COW", "SCHL", "MAR", "OCCP", "POBP", "SEX", "RAC1P"};

            for (String qid : others) {
                AttributeType.Hierarchy hierarchy = AttributeType.Hierarchy.create(hierarchyPath + qid + ".csv", StandardCharsets.UTF_8, ';');
                data.getDefinition().setHierarchy(qid, hierarchy);
            }
        }else{
            // we will read previously generated hierarchy files
            String hierarchyPath = "../Hierarchies/ACSIncome/";
            for (String qid : data.getDefinition().getQuasiIdentifyingAttributes()) {
                AttributeType.Hierarchy hierarchy = AttributeType.Hierarchy.create(hierarchyPath + qid + ".csv", StandardCharsets.UTF_8, ';');
                data.getDefinition().setHierarchy(qid, hierarchy);
            }
        }

        return data;
    }

    public Map<String,double[]> getParams(){
        Map<String,double[]> configParams = new LinkedHashMap<>();
        configParams.put("k", new double[]{5.0, 10.0, 20.0});
        return configParams;
    }
}
