import io.fabric8.kubernetes.api.model.apps.Deployment;
import io.fabric8.kubernetes.client.DefaultKubernetesClient;
import io.fabric8.kubernetes.client.KubernetesClient;
import java.time.Instant;

public class PodRestarter {

    public static void restartDeployment(String namespace, String deploymentName) {
        try (KubernetesClient client = new DefaultKubernetesClient()) {
            Deployment deployment = client.apps().deployments()
                    .inNamespace(namespace)
                    .withName(deploymentName)
                    .edit(d -> new DeploymentBuilder(d)
                            .editSpec()
                                .editTemplate()
                                    .editMetadata()
                                        .addToAnnotations("kubectl.kubernetes.io/restartedAt", Instant.now().toString())
                                    .endMetadata()
                                .endTemplate()
                            .endSpec()
                            .build());

            client.apps().deployments()
                    .inNamespace(namespace)
                    .withName(deploymentName)
                    .replace(deployment);

            System.out.println("Deployment restarted successfully.");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

<dependency>
    <groupId>io.fabric8</groupId>
    <artifactId>kubernetes-client</artifactId>
    <version>6.9.1</version> <!-- Use latest stable -->
</dependency>
