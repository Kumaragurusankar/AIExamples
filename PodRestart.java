import io.fabric8.kubernetes.client.KubernetesClient;
import io.fabric8.kubernetes.client.KubernetesClientBuilder;
import io.fabric8.openshift.client.OpenShiftClient;

public class OpenShiftPodManager {

    public static void restartDeployment(String namespace, String deploymentName) {
        try (KubernetesClient client = new KubernetesClientBuilder().build()) {
            OpenShiftClient oClient = client.adapt(OpenShiftClient.class);

            // Get current replica count
            int currentReplicas = oClient.deploymentConfigs()
                .inNamespace(namespace)
                .withName(deploymentName)
                .get()
                .getSpec()
                .getReplicas();

            System.out.println("Current replica count: " + currentReplicas);

            // Scale to 0 (stop)
            oClient.deploymentConfigs()
                .inNamespace(namespace)
                .withName(deploymentName)
                .scale(0, true); // Wait until scaled

            System.out.println("Scaled down to 0");

            // Optional: wait before restarting
            Thread.sleep(3000);

            // Scale back to original
            oClient.deploymentConfigs()
                .inNamespace(namespace)
                .withName(deploymentName)
                .scale(currentReplicas, true);

            System.out.println("Scaled back to " + currentReplicas);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
