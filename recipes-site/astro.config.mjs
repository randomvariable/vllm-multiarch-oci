import starlight from "@astrojs/starlight";
import { defineConfig } from "astro/config";

export default defineConfig({
  site: "https://randomvariable.github.io",
  base: "/vllm-multiarch-oci",
  output: "static",
  trailingSlash: "always",
  build: { format: "directory" },
  integrations: [
    starlight({
      title: "vLLM multi-arch recipes",
      description: "Reproducible vLLM deployments for consumer Blackwell hardware.",
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/randomvariable/vllm-multiarch-oci",
        },
      ],
      sidebar: [
        {
          label: "Recipes",
          items: [
            { label: "Catalog", link: "/recipes/" },
            {
              label: "DeepSeek V4 Flash Vision",
              link: "/recipes/deepseek-v4-flash-vision-tp2/",
            },
          ],
        },
        { label: "Guides", items: [{ autogenerate: { directory: "guides" } }] },
        {
          label: "Helpers",
          items: [
            {
              label: "Kubernetes image helpers",
              link: "/helpers/kubernetes-image-helpers/",
            },
          ],
        },
        {
          label: "Image releases",
          items: [{ label: "Latest image", link: "/image-releases/" }],
        },
      ],
      customCss: ["./src/styles/recipes.css"],
      pagination: false,
      editLink: {
        baseUrl:
          "https://github.com/randomvariable/vllm-multiarch-oci/edit/main/recipes-site/src/content/docs/",
      },
      lastUpdated: false,
    }),
  ],
});
