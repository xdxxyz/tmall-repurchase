package com.nju.tmall;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import scala.Tuple2;

import java.util.*;

public class Task3_KMeans {
    // PPT要求固定为3维特征: F1_norm, F2_norm, F3_norm
    private static final int VECTOR_SIZE = 3;

    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println("Usage: Task3_KMeans <input_path> <output_path> <k>");
            System.exit(1);
        }
        String inputPath = args[0];   // user_features.csv 所在路径
        String outputPath = args[1];  // 例如 /user/231220104a/task3_out
        int k = Integer.parseInt(args[2]);

        SparkConf conf = new SparkConf().setAppName("Task3_KMeans");
        JavaSparkContext sc = new JavaSparkContext(conf);

        /* ===================== 1. 读取任务1输出的3维特征 ===================== */
        // 输入格式: user_id,F1_norm,F2_norm,F3_norm
        JavaRDD<Tuple2<String, double[]>> userData = sc.textFile(inputPath)
                .filter(line -> line != null && !line.trim().isEmpty())
                .map(line -> {
                    String[] parts = line.split(",");
                    String uid = parts[0].trim();
                    double[] vec = new double[VECTOR_SIZE];
                    for (int i = 0; i < VECTOR_SIZE; i++) {
                        vec[i] = Double.parseDouble(parts[i + 1].trim());
                    }
                    return new Tuple2<>(uid, vec);
                });

        /* ===================== 2. 随机抽样初始化 k 个聚类中心 ===================== */
        List<Tuple2<String, double[]>> allUsers = userData.collect();
        if (allUsers.size() < k) {
            System.err.println("Error: users=" + allUsers.size() + " < k=" + k);
            sc.close();
            System.exit(1);
        }

        Random rand = new Random(42);
        List<double[]> centers = new ArrayList<>();
        Set<Integer> picked = new HashSet<>();
        while (centers.size() < k) {
            int idx = rand.nextInt(allUsers.size());
            if (picked.add(idx)) {
                centers.add(allUsers.get(idx)._2().clone());
            }
        }

        /* ===================== 3. KMeans 迭代 (使用广播变量) ===================== */
        boolean converged = false;
        for (int iter = 0; iter < 30 && !converged; iter++) {
            System.out.println("Iteration " + (iter + 1));
            Broadcast<List<double[]>> bcCenters = sc.broadcast(centers);

            // E-step: 分配类簇
            JavaRDD<Tuple2<Integer, double[]>> assigned = userData.map(user -> {
                double[] point = user._2();
                List<double[]> cList = bcCenters.value();
                double bestDist = Double.MAX_VALUE;
                int bestCluster = 0;
                for (int i = 0; i < cList.size(); i++) {
                    double dist = euclideanDistance(point, cList.get(i));
                    if (dist < bestDist) {
                        bestDist = dist;
                        bestCluster = i;
                    }
                }
                return new Tuple2<>(bestCluster, point);
            });

            // M-step: 聚合更新中心点
            Map<Integer, Tuple2<double[], Integer>> clusterSums = assigned
                    .mapToPair(t -> new Tuple2<>(t._1(), new Tuple2<>(t._2(), 1)))
                    .reduceByKey((v1, v2) -> {
                        double[] sum = new double[VECTOR_SIZE];
                        for (int i = 0; i < VECTOR_SIZE; i++) {
                            sum[i] = v1._1()[i] + v2._1()[i];
                        }
                        return new Tuple2<>(sum, v1._2() + v2._2());
                    })
                    .collectAsMap();

            List<double[]> newCenters = new ArrayList<>(Collections.nCopies(k, null));
            double totalShift = 0.0;

            for (int ci = 0; ci < k; ci++) {
                Tuple2<double[], Integer> sumCount = clusterSums.get(ci);
                double[] newCenter;
                if (sumCount != null && sumCount._2() > 0) {
                    newCenter = new double[VECTOR_SIZE];
                    for (int i = 0; i < VECTOR_SIZE; i++) {
                        newCenter[i] = sumCount._1()[i] / sumCount._2();
                    }
                } else {
                    // 防止空簇：保留原中心
                    newCenter = centers.get(ci).clone();
                }
                newCenters.set(ci, newCenter);
                totalShift += euclideanDistance(centers.get(ci), newCenter);
            }

            if (totalShift < 1e-6) {
                converged = true;
                System.out.println("Converged at iteration " + (iter + 1));
            } else {
                centers = newCenters;
            }
            bcCenters.unpersist();
        }

        /* ===================== 4. 生成最终标签 ===================== */
        Broadcast<List<double[]>> finalBc = sc.broadcast(centers);
        JavaRDD<String> labelLines = userData.map(user -> {
            double[] point = user._2();
            List<double[]> cList = finalBc.value();
            double bestDist = Double.MAX_VALUE;
            int bestCluster = 0;
            for (int i = 0; i < cList.size(); i++) {
                double dist = euclideanDistance(point, cList.get(i));
                if (dist < bestDist) {
                    bestDist = dist;
                    bestCluster = i;
                }
            }
            return user._1() + "," + bestCluster;
        });

        /* ===================== 5. 按PPT格式输出 ===================== */
        // cluster_centers.txt: cluster_id\tF1,F2,F3 (逗号分隔)
        List<String> centerStrs = new ArrayList<>();
        for (int i = 0; i < centers.size(); i++) {
            double[] c = centers.get(i);
            centerStrs.add(i + "\t" + String.format(Locale.US, "%.4f,%.4f,%.4f", c[0], c[1], c[2]));
        }
        JavaRDD<String> centerLines = sc.parallelize(centerStrs);

        Configuration hcfg = sc.hadoopConfiguration();
        FileSystem fs = FileSystem.get(hcfg);
        Path outBase = new Path(outputPath);

        if (fs.exists(outBase)) fs.delete(outBase, true);

        // 保存 cluster_centers.txt
        Path tmpC = new Path(outputPath + "_tmp_centers");
        if (fs.exists(tmpC)) fs.delete(tmpC, true);
        centerLines.coalesce(1).saveAsTextFile(tmpC.toString());
        movePartToTarget(fs, tmpC, new Path(outputPath, "cluster_centers.txt"));
        fs.delete(tmpC, true);

        // 保存 user_cluster_labels.csv
        Path tmpL = new Path(outputPath + "_tmp_labels");
        if (fs.exists(tmpL)) fs.delete(tmpL, true);
        labelLines.coalesce(1).saveAsTextFile(tmpL.toString());
        movePartToTarget(fs, tmpL, new Path(outputPath, "user_cluster_labels.csv"));
        fs.delete(tmpL, true);

        System.out.println("==== TASK3 DONE ====");
        System.out.println("Outputs:");
        System.out.println("  cluster_centers.txt     : " + new Path(outputPath, "cluster_centers.txt"));
        System.out.println("  user_cluster_labels.csv : " + new Path(outputPath, "user_cluster_labels.csv"));
        sc.close();
    }

    private static void movePartToTarget(FileSystem fs, Path tmpDir, Path target) throws Exception {
        Path part = new Path(tmpDir, "part-00000");
        if (!fs.exists(part)) {
            boolean found = false;
            if (fs.exists(tmpDir)) {
                for (FileStatus s : fs.listStatus(tmpDir)) {
                    if (s.isFile() && s.getPath().getName().startsWith("part-")) {
                        part = s.getPath();
                        found = true;
                        break;
                    }
                }
            }
            if (!found) throw new IllegalStateException("找不到 part 文件在: " + tmpDir);
        }
        if (!fs.exists(target.getParent())) fs.mkdirs(target.getParent());
        if (fs.exists(target)) fs.delete(target, false);
        boolean ok = fs.rename(part, target);
        if (!ok) {
            throw new RuntimeException("HDFS rename 失败：" + part + " -> " + target);
        }
    }

    private static double euclideanDistance(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }
}
