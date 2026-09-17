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
      title: "local-inference-lab/vLLM for DGX Spark",
      description: "Build and deployment recipes for the local-inference-lab vLLM fork on NVIDIA DGX Spark.",
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
            {
              label: "Qwen3.8 Flash Next",
              link: "/recipes/qwen38-flash-next-tp2/",
            },
          ],
        },
        { label: "Guides", items: [{ autogenerate: { directory: "guides" } }] },
        {
          label: "Explanations",
          items: [{ autogenerate: { directory: "explanation" } }],
        },
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
          items: [
            { label: "Latest image", link: "/image-releases/" },
            {
              label: "Included upstream changes",
              link: "/image-releases/included-changes/",
            },
          ],
        },
        { label: "About the author", link: "/about/" },
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
