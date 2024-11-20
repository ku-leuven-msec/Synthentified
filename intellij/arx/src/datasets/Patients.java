package datasets;

import org.deidentifier.arx.*;
import org.deidentifier.arx.aggregates.AggregateFunction;
import org.deidentifier.arx.aggregates.HierarchyBuilderIntervalBased;
import org.deidentifier.arx.aggregates.HierarchyBuilderOrderBased;
import org.deidentifier.arx.aggregates.HierarchyBuilderRedactionBased;
import org.deidentifier.arx.criteria.KAnonymity;
import org.deidentifier.arx.metric.Metric;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.stream.IntStream;

public class Patients extends Dataset {

    public Patients(String input, String output, String statisticsOutput){
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
        source.addColumn("gender", DataType.STRING, true);
        source.addColumn("age", DataType.INTEGER, true);
        source.addColumn("hypertension", DataType.INTEGER, true);
        source.addColumn("heart_disease", DataType.INTEGER, true);
        source.addColumn("avg_glucose_level", DataType.DECIMAL, true);
        source.addColumn("bmi", DataType.DECIMAL, true);
        source.addColumn("smoking_status", DataType.createOrderedString(new String[]{"never smoked", "formerly smoked", "smokes"}), true);
        source.addColumn("stroke", DataType.INTEGER, true);
        source.addColumn("Income", DataType.INTEGER, true);
        source.addColumn("ZIP", DataType.STRING, true);

        Data data = Data.create(source);

        data.getDefinition().setAttributeType("gender", AttributeType.QUASI_IDENTIFYING_ATTRIBUTE);
        data.getDefinition().setAttributeType("age", AttributeType.QUASI_IDENTIFYING_ATTRIBUTE);
        data.getDefinition().setAttributeType("smoking_status", AttributeType.QUASI_IDENTIFYING_ATTRIBUTE);
        data.getDefinition().setAttributeType("ZIP", AttributeType.QUASI_IDENTIFYING_ATTRIBUTE);

        // are sensitive but I am not applying any privacy model on them (must be set on insensitive or we get errors)
        data.getDefinition().setAttributeType("bmi", AttributeType.INSENSITIVE_ATTRIBUTE);
        data.getDefinition().setAttributeType("hypertension", AttributeType.INSENSITIVE_ATTRIBUTE);
        data.getDefinition().setAttributeType("heart_disease", AttributeType.INSENSITIVE_ATTRIBUTE);
        data.getDefinition().setAttributeType("avg_glucose_level", AttributeType.INSENSITIVE_ATTRIBUTE);
        data.getDefinition().setAttributeType("stroke", AttributeType.INSENSITIVE_ATTRIBUTE);
        data.getDefinition().setAttributeType("Income", AttributeType.INSENSITIVE_ATTRIBUTE);

        if(generateHierarchies) {
            String hierarchyPath = "../Hierarchies/patients/";
            File dir = new File(hierarchyPath);
            dir.mkdirs();
            // gender
            {
                HierarchyBuilderOrderBased<String> builderOrderBased = HierarchyBuilderOrderBased.create(DataType.STRING, false);
                String[] tmp = data.getHandle().getDistinctValues(data.getHandle().getColumnIndexOf("gender")).clone();
                Arrays.sort(tmp);
                AttributeType.Hierarchy hierarchy = builderOrderBased.build(tmp);
                data.getDefinition().setHierarchy("gender", hierarchy);
                hierarchy.save(hierarchyPath + "gender.csv");
            }

            // age
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
                data.getDefinition().setHierarchy("age", hierarchy);
                hierarchy.save(hierarchyPath + "age.csv");
            }

            // ZIP
            {
                HierarchyBuilderRedactionBased<String> builderRedactionBased = HierarchyBuilderRedactionBased.create(HierarchyBuilderRedactionBased.Order.LEFT_TO_RIGHT, HierarchyBuilderRedactionBased.Order.RIGHT_TO_LEFT, ' ', '*');
                builderRedactionBased.setDomainAndAlphabetSize(13879, 10, 5);

                String[] tmp = data.getHandle().getDistinctValues(data.getHandle().getColumnIndexOf("ZIP")).clone();
                Arrays.sort(tmp);

                AttributeType.Hierarchy hierarchy = builderRedactionBased.build(tmp);
                data.getDefinition().setHierarchy("ZIP", hierarchy);
                hierarchy.save(hierarchyPath+"ZIP.csv");
            }

            // smoking_status
            {
                HierarchyBuilderOrderBased<String> builderOrderBased = HierarchyBuilderOrderBased.create(DataType.STRING, new String[]{"never smoked", "formerly smoked", "smokes"});
                builderOrderBased.getLevel(0).addGroup(1, "never smoked");
                builderOrderBased.getLevel(0).addGroup(2, "ever smoked");

                String[] tmp = data.getHandle().getDistinctValues(data.getHandle().getColumnIndexOf("smoking_status")).clone();
                Arrays.sort(tmp);
                AttributeType.Hierarchy hierarchy = builderOrderBased.build(tmp);
                data.getDefinition().setHierarchy("smoking_status", hierarchy);
                hierarchy.save(hierarchyPath+"smoking_status.csv");
            }
        }else{
            // we will read previously generated hierarchy files
            String hierarchyPath = "../Hierarchies/patients/";
            for(String qid:data.getDefinition().getQuasiIdentifyingAttributes()){
                AttributeType.Hierarchy hierarchy = AttributeType.Hierarchy.create(hierarchyPath + qid + ".csv",StandardCharsets.UTF_8,';');
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
