import io.fabric8.kubernetes.client.KubernetesClient;
import io.fabric8.kubernetes.client.KubernetesClientBuilder;
import io.fabric8.kubernetes.api.model.apps.Deployment;
import io.fabric8.openshift.client.OpenShiftClient;

public class DeploymentRestarter {

    public static void restart(String namespace, String name, boolean isDeploymentConfig) {
        try (KubernetesClient client = new KubernetesClientBuilder().build()) {
            OpenShiftClient openShiftClient = client.adapt(OpenShiftClient.class);

            if (isDeploymentConfig) {
                // OpenShift DeploymentConfig
                var dc = openShiftClient.deploymentConfigs()
                        .inNamespace(namespace)
                        .withName(name)
                        .get();

                if (dc == null) {
                    System.out.println("DeploymentConfig not found.");
                    return;
                }

                int replicas = dc.getSpec().getReplicas();
                System.out.println("Restarting DeploymentConfig: " + name + " with replicas: " + replicas);

                // Scale to 0
                openShiftClient.deploymentConfigs()
                        .inNamespace(namespace)
                        .withName(name)
                        .scale(0, true);

                // Wait before scaling up
                Thread.sleep(3000);

                // Scale back to original
                openShiftClient.deploymentConfigs()
                        .inNamespace(namespace)
                        .withName(name)
                        .scale(replicas, true);

                System.out.println("Restarted DeploymentConfig: " + name);

            } else {
                // Kubernetes Deployment
                var deployment = client.apps()
                        .deployments()
                        .inNamespace(namespace)
                        .withName(name)
                        .get();

                if (deployment == null) {
                    System.out.println("Deployment not found.");
                    return;
                }

                int replicas = deployment.getSpec().getReplicas();
                System.out.println("Restarting Deployment: " + name + " with replicas: " + replicas);

                // Scale to 0
                client.apps()
                        .deployments()
                        .inNamespace(namespace)
                        .withName(name)
                        .scale(0, true);

                Thread.sleep(3000);

                // Scale back
                client.apps()
                        .deployments()
                        .inNamespace(namespace)
                        .withName(name)
                        .scale(replicas, true);

                System.out.println("Restarted Deployment: " + name);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
