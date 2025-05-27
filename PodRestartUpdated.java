try (KubernetesClient client = new KubernetesClientBuilder().build()) {
            OpenShiftClient openShiftClient = client.adapt(OpenShiftClient.class);

            if (isDeploymentConfig) {
                // 🔁 Restart OpenShift DeploymentConfig
                DeploymentConfig dc = openShiftClient.deploymentConfigs()
                        .inNamespace(namespace)
                        .withName(name)
                        .get();

                if (dc == null) {
                    System.err.println("DeploymentConfig not found: " + name);
                    return;
                }

                int originalReplicas = dc.getSpec().getReplicas();

                // Scale down
                openShiftClient.deploymentConfigs()
                        .inNamespace(namespace)
                        .withName(name)
                        .edit(d -> {
                            d.getSpec().setReplicas(0);
                            return d;
                        });

                Thread.sleep(3000); // Optional wait

                // Scale up
                openShiftClient.deploymentConfigs()
                        .inNamespace(namespace)
                        .withName(name)
                        .edit(d -> {
                            d.getSpec().setReplicas(originalReplicas);
                            return d;
                        });

                System.out.println("✅ Restarted DeploymentConfig: " + name);

            } else {
                // 🔁 Restart Kubernetes Deployment
                Deployment deployment = client.apps()
                        .deployments()
                        .inNamespace(namespace)
                        .withName(name)
                        .get();

                if (deployment == null) {
                    System.err.println("Deployment not found: " + name);
                    return;
                }

                int originalReplicas = deployment.getSpec().getReplicas();

                // Scale down
                client.apps()
                        .deployments()
                        .inNamespace(namespace)
                        .withName(name)
                        .edit(d -> {
                            d.getSpec().setReplicas(0);
                            return d;
                        });

                Thread.sleep(3000); // Optional wait

                // Scale back up
                client.apps()
                        .deployments()
                        .inNamespace(namespace)
                        .withName(name)
                        .edit(d -> {
                            d.getSpec().setReplicas(originalReplicas);
                            return d;
                        });

                System.out.println("✅ Restarted Deployment: " + name);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
