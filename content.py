"""
Hand-written text for the overview pages, the About page and the comparisons.

Everything here is plain data: edit the wording freely. Keys must match the
values used in the CSVs exactly (codename, family, GPU vendor). Entries for
names that are not in the data are simply ignored, and pages for groups with
no entry here fall back to the sentences generated from the data.

"facts" are platform-level facts for a whole generation (the maximum the
platform supports), not the range of models listed in the database.
"""

# ---------- CPU generations (keyed by the CSV "Codename" value) ----------

GENERATIONS: dict[str, dict] = {
    # ----- AMD EPYC -----
    "Naples": {
        "intro": [
            "Naples is the first generation of AMD EPYC, sold as the EPYC 7001 series from June 2017. "
            "It brought AMD back to the server market with the Zen core and a multi-chip design that "
            "places four 8-core dies in one package.",
            "Every Naples processor gets the full platform: eight DDR4 memory channels and 128 PCIe 3.0 "
            "lanes on the SP3 socket, even in single-socket servers.",
        ],
        "facts": [
            ("Product line", "AMD EPYC 7001 series"), ("Launched", "June 2017"), ("Architecture", "Zen"),
            ("Process", "14 nm"), ("Socket", "SP3"), ("Max cores", "32"),
            ("Memory", "8 channels DDR4-2666"), ("PCIe", "128 lanes PCIe 3.0"),
        ],
    },
    "Rome": {
        "intro": [
            "Rome is the second generation of AMD EPYC, sold as the EPYC 7002 series from August 2019. "
            "It introduced AMD's chiplet layout for servers: up to eight 7 nm compute dies (CCDs) "
            "around a central 14 nm I/O die.",
            "Rome doubled the core count to 64 per socket and was the first x86 server processor with "
            "PCIe 4.0, while staying compatible with the SP3 socket used by Naples.",
        ],
        "facts": [
            ("Product line", "AMD EPYC 7002 series"), ("Launched", "August 2019"), ("Architecture", "Zen 2"),
            ("Process", "7 nm compute dies, 14 nm I/O die"), ("Socket", "SP3"), ("Max cores", "64"),
            ("Memory", "8 channels DDR4-3200"), ("PCIe", "128 lanes PCIe 4.0"),
        ],
    },
    "Milan": {
        "intro": [
            "Milan is the third generation of AMD EPYC, sold as the EPYC 7003 series from March 2021. "
            "It moved to the Zen 3 core, in which all eight cores of a compute die share one 32 MB L3 "
            "cache, a large gain for latency-sensitive and per-core-licensed workloads.",
            "Milan kept Rome's SP3 socket, 64-core maximum, eight DDR4 channels and 128 PCIe 4.0 lanes, "
            "so many servers could be upgraded with a BIOS update.",
        ],
        "facts": [
            ("Product line", "AMD EPYC 7003 series"), ("Launched", "March 2021"), ("Architecture", "Zen 3"),
            ("Process", "7 nm compute dies"), ("Socket", "SP3"), ("Max cores", "64"),
            ("Memory", "8 channels DDR4-3200"), ("PCIe", "128 lanes PCIe 4.0"),
        ],
    },
    "Milan-X": {
        "intro": [
            "Milan-X is a version of Milan with AMD 3D V-Cache, launched in March 2022. Extra SRAM is "
            "stacked on top of each compute die, which triples the L3 cache to as much as 768 MB per "
            "socket.",
            "It targets technical computing such as CFD, finite element analysis and EDA, where large "
            "working sets fit in cache. Milan-X uses the same SP3 platform as Milan.",
        ],
        "facts": [
            ("Product line", "AMD EPYC 7003 series with 3D V-Cache"), ("Launched", "March 2022"),
            ("Architecture", "Zen 3 with 3D V-Cache"), ("Socket", "SP3"), ("Max cores", "64"),
            ("Max L3 cache", "768 MB"), ("Memory", "8 channels DDR4-3200"), ("PCIe", "128 lanes PCIe 4.0"),
        ],
    },
    "Genoa": {
        "intro": [
            "Genoa is the fourth generation of AMD EPYC, sold as the EPYC 9004 series from November 2022. "
            "It moved to the Zen 4 core and a new platform, the SP5 socket, with up to 96 cores per socket.",
            "Genoa brought DDR5 memory with twelve channels, 128 lanes of PCIe 5.0, CXL memory expansion "
            "and, for the first time on EPYC, AVX-512 instructions.",
        ],
        "facts": [
            ("Product line", "AMD EPYC 9004 series"), ("Launched", "November 2022"), ("Architecture", "Zen 4"),
            ("Process", "5 nm compute dies, 6 nm I/O die"), ("Socket", "SP5"), ("Max cores", "96"),
            ("Memory", "12 channels DDR5-4800"), ("PCIe", "128 lanes PCIe 5.0, CXL 1.1+"),
        ],
    },
    "Genoa-X": {
        "intro": [
            "Genoa-X is a version of Genoa with AMD 3D V-Cache, launched in June 2023. Stacked cache "
            "raises the L3 to as much as 1,152 MB per socket.",
            "Like Milan-X before it, Genoa-X is aimed at technical computing workloads that benefit from "
            "very large caches. It uses the same SP5 platform as Genoa.",
        ],
        "facts": [
            ("Product line", "AMD EPYC 9004 series with 3D V-Cache"), ("Launched", "June 2023"),
            ("Architecture", "Zen 4 with 3D V-Cache"), ("Socket", "SP5"), ("Max cores", "96"),
            ("Max L3 cache", "1,152 MB"), ("Memory", "12 channels DDR5-4800"), ("PCIe", "128 lanes PCIe 5.0"),
        ],
    },
    "Bergamo": {
        "intro": [
            "Bergamo is the cloud-native member of the EPYC 9004 family, launched in June 2023. It uses "
            "Zen 4c, a compact version of the Zen 4 core with less cache per core, to fit up to 128 cores "
            "in one SP5 socket.",
            "Bergamo targets workloads that scale with core count, such as web serving, containers and "
            "microservices, rather than per-core speed.",
        ],
        "facts": [
            ("Product line", "AMD EPYC 9004 series (97x4)"), ("Launched", "June 2023"),
            ("Architecture", "Zen 4c"), ("Socket", "SP5"), ("Max cores", "128"),
            ("Memory", "12 channels DDR5-4800"), ("PCIe", "128 lanes PCIe 5.0"),
        ],
    },
    "Siena": {
        "intro": [
            "Siena is the AMD EPYC 8004 series, launched in September 2023 for edge, telco and other "
            "space- and power-constrained servers. It uses Zen 4c cores on a smaller, cheaper platform, "
            "the SP6 socket.",
            "Siena offers up to 64 cores, six DDR5 channels and 96 PCIe 5.0 lanes at lower power than "
            "the SP5 parts.",
        ],
        "facts": [
            ("Product line", "AMD EPYC 8004 series"), ("Launched", "September 2023"), ("Architecture", "Zen 4c"),
            ("Socket", "SP6"), ("Max cores", "64"), ("Memory", "6 channels DDR5-4800"),
            ("PCIe", "96 lanes PCIe 5.0"),
        ],
    },
    "Turin": {
        "intro": [
            "Turin is the fifth generation of AMD EPYC, sold as the EPYC 9005 series from October 2024. "
            "It comes in two versions on the same SP5 socket as Genoa: Zen 5 models with up to 128 cores, "
            "and Zen 5c models with up to 192 cores.",
            "Turin raises memory speed to DDR5-6000 across twelve channels and keeps 128 PCIe 5.0 lanes, "
            "with support for CXL 2.0.",
        ],
        "facts": [
            ("Product line", "AMD EPYC 9005 series"), ("Launched", "October 2024"),
            ("Architecture", "Zen 5 and Zen 5c"), ("Socket", "SP5"), ("Max cores", "192 (Zen 5c), 128 (Zen 5)"),
            ("Memory", "12 channels DDR5-6000"), ("PCIe", "128 lanes PCIe 5.0, CXL 2.0"),
        ],
    },
    "Raphael": {
        "intro": [
            "Raphael is the codename of AMD's Ryzen 7000 desktop silicon, which AMD also sells for entry "
            "servers as the EPYC 4004 series, launched in May 2024. These processors use the Zen 4 core on "
            "the AM5 socket.",
            "The EPYC 4004 series targets small business and hosting servers: up to 16 cores, two channels "
            "of ECC DDR5 memory and 28 PCIe 5.0 lanes, on lower-cost motherboards.",
        ],
        "facts": [
            ("Product line", "AMD EPYC 4004 series"), ("Launched", "May 2024"), ("Architecture", "Zen 4"),
            ("Socket", "AM5"), ("Max cores", "16"), ("Memory", "2 channels DDR5, up to 192 GB"),
            ("PCIe", "28 lanes PCIe 5.0"),
        ],
    },

    # ----- Intel Xeon Scalable and Xeon 6 -----
    "Skylake": {
        "intro": [
            "Skylake-SP is the first generation of Intel Xeon Scalable processors, launched in July 2017. "
            "It replaced the Xeon E5 and E7 lines with the Platinum, Gold, Silver and Bronze tiers still "
            "used today.",
            "Skylake-SP introduced a mesh interconnect between cores, AVX-512 instructions and six DDR4 "
            "memory channels per socket on the LGA 3647 socket.",
        ],
        "facts": [
            ("Product line", "1st Gen Intel Xeon Scalable"), ("Launched", "July 2017"), ("Process", "14 nm"),
            ("Socket", "LGA 3647"), ("Max cores", "28"), ("Memory", "6 channels DDR4-2666"),
            ("PCIe", "48 lanes PCIe 3.0"),
        ],
    },
    "Cascade Lake": {
        "intro": [
            "Cascade Lake is the second generation of Intel Xeon Scalable processors, launched in April 2019, "
            "with a refresh of cheaper, higher-clocked models (the \"R\" parts) in February 2020.",
            "It kept the Skylake-SP platform and core counts, and added DL Boost (VNNI) instructions for AI "
            "inference, faster DDR4-2933 memory, support for Intel Optane persistent memory and hardware "
            "fixes for several Spectre and Meltdown variants.",
        ],
        "facts": [
            ("Product line", "2nd Gen Intel Xeon Scalable"), ("Launched", "April 2019"), ("Process", "14 nm"),
            ("Socket", "LGA 3647"), ("Max cores", "28 (56 in the Platinum 9200 series)"),
            ("Memory", "6 channels DDR4-2933"), ("PCIe", "48 lanes PCIe 3.0"),
        ],
    },
    "Ice Lake": {
        "intro": [
            "Ice Lake-SP is the third generation of Intel Xeon Scalable processors for one- and two-socket "
            "servers, launched in April 2021. It was Intel's first 10 nm server processor.",
            "Ice Lake raised the maximum to 40 cores and added eight DDR4-3200 memory channels and PCIe 4.0 "
            "on the new LGA 4189 socket.",
        ],
        "facts": [
            ("Product line", "3rd Gen Intel Xeon Scalable"), ("Launched", "April 2021"), ("Process", "10 nm"),
            ("Socket", "LGA 4189"), ("Max cores", "40"), ("Memory", "8 channels DDR4-3200"),
            ("PCIe", "64 lanes PCIe 4.0"),
        ],
    },
    "Sapphire Rapids": {
        "intro": [
            "Sapphire Rapids is the fourth generation of Intel Xeon Scalable processors, launched in "
            "January 2023. The larger models are built from four tiles connected with Intel's EMIB "
            "packaging.",
            "It brought DDR5, PCIe 5.0 and CXL to Xeon, along with Advanced Matrix Extensions (AMX) for AI "
            "and built-in accelerators for compression, encryption and data movement (QAT, DSA, IAA, DLB), "
            "which are enabled on selected models.",
        ],
        "facts": [
            ("Product line", "4th Gen Intel Xeon Scalable"), ("Launched", "January 2023"),
            ("Process", "Intel 7"), ("Socket", "LGA 4677"), ("Max cores", "60"),
            ("Memory", "8 channels DDR5-4800"), ("PCIe", "80 lanes PCIe 5.0, CXL 1.1"),
        ],
    },
    "Sapphire Rapids HBM": {
        "intro": [
            "The Intel Xeon CPU Max series, launched in January 2023, is Sapphire Rapids with 64 GB of "
            "HBM2e memory inside the package. It is aimed at HPC codes that are limited by memory bandwidth.",
            "The HBM can act as a cache in front of DDR5, as a separate memory region, or as the only memory "
            "in the system.",
        ],
        "facts": [
            ("Product line", "Intel Xeon CPU Max series"), ("Launched", "January 2023"),
            ("Process", "Intel 7"), ("Socket", "LGA 4677"), ("Max cores", "56"),
            ("On-package memory", "64 GB HBM2e"), ("Memory", "8 channels DDR5-4800"),
        ],
    },
    "Emerald Rapids": {
        "intro": [
            "Emerald Rapids is the fifth generation of Intel Xeon Scalable processors, launched in "
            "December 2023. It is a drop-in upgrade for Sapphire Rapids servers on the same LGA 4677 socket.",
            "Emerald Rapids uses two larger tiles instead of four, which allowed up to 64 cores, a much larger "
            "L3 cache of up to 320 MB and faster DDR5-5600 memory.",
        ],
        "facts": [
            ("Product line", "5th Gen Intel Xeon Scalable"), ("Launched", "December 2023"),
            ("Process", "Intel 7"), ("Socket", "LGA 4677"), ("Max cores", "64"), ("Max L3 cache", "320 MB"),
            ("Memory", "8 channels DDR5-5600"), ("PCIe", "80 lanes PCIe 5.0"),
        ],
    },
    "Granite Rapids": {
        "intro": [
            "Granite Rapids is the performance-core (P-core) half of the Intel Xeon 6 family. The top Xeon "
            "6900P models launched in September 2024 with up to 128 cores, and the mainstream Xeon 6700P "
            "and 6500P models followed in February 2025.",
            "Granite Rapids is built on the Intel 3 process. The 6900P platform supports twelve channels of "
            "DDR5-6400 or faster MRDIMM memory, while the 6700P platform uses eight channels.",
        ],
        "facts": [
            ("Product line", "Intel Xeon 6 with P-cores"), ("Launched", "September 2024 (6900P), February 2025 (6700P/6500P)"),
            ("Process", "Intel 3"), ("Socket", "LGA 7529 (6900P), LGA 4710 (6700P/6500P)"), ("Max cores", "128"),
            ("Memory", "12 channels DDR5-6400 or MRDIMM-8800 (6900P)"),
        ],
    },
    "Sierra Forest": {
        "intro": [
            "Sierra Forest is the efficiency-core (E-core) half of the Intel Xeon 6 family, launched in "
            "June 2024 as the Xeon 6700E series. It is Intel's first Xeon built only from E-cores.",
            "Sierra Forest packs up to 144 single-threaded cores per socket for highly parallel cloud and "
            "networking workloads. The E-cores do not support AVX-512 or AMX.",
        ],
        "facts": [
            ("Product line", "Intel Xeon 6 with E-cores"), ("Launched", "June 2024"), ("Process", "Intel 3"),
            ("Socket", "LGA 4710"), ("Max cores", "144"), ("Threads per core", "1"),
            ("Memory", "8 channels DDR5-6400"),
        ],
    },

    # ----- Intel Xeon E5 and older -----
    "Broadwell": {
        "intro": [
            "Broadwell is Intel's 14 nm shrink of the Haswell design. In two-socket servers it was sold as the "
            "Xeon E5-2600 v4 series (Broadwell-EP), launched in March 2016.",
            "Broadwell-EP raised the maximum to 22 cores on the LGA 2011-3 socket used by Haswell-EP, with "
            "four channels of DDR4-2400.",
        ],
        "facts": [
            ("Product line", "Intel Xeon E5 v4"), ("Launched", "March 2016"), ("Process", "14 nm"),
            ("Socket", "LGA 2011-3"), ("Max cores", "22"), ("Memory", "4 channels DDR4-2400"),
            ("PCIe", "40 lanes PCIe 3.0"),
        ],
    },
    "Haswell": {
        "intro": [
            "Haswell-EP is the Xeon E5-2600 v3 series, launched in September 2014. It was the first Xeon "
            "platform with DDR4 memory and moved servers to the new LGA 2011-3 socket.",
            "Haswell added AVX2 and FMA instructions, which roughly doubled peak floating-point throughput "
            "per core, and offered up to 18 cores.",
        ],
        "facts": [
            ("Product line", "Intel Xeon E5 v3"), ("Launched", "September 2014"), ("Process", "22 nm"),
            ("Socket", "LGA 2011-3"), ("Max cores", "18"), ("Memory", "4 channels DDR4-2133"),
            ("PCIe", "40 lanes PCIe 3.0"),
        ],
    },
    "Ivy Bridge-EP": {
        "intro": [
            "Ivy Bridge-EP is the Xeon E5-2600 v2 series, launched in September 2013. It is the 22 nm shrink "
            "of Sandy Bridge-EP and uses the same LGA 2011 socket.",
            "It raised the maximum to 12 cores and supported faster DDR3-1866 memory.",
        ],
        "facts": [
            ("Product line", "Intel Xeon E5 v2"), ("Launched", "September 2013"), ("Process", "22 nm"),
            ("Socket", "LGA 2011"), ("Max cores", "12"), ("Memory", "4 channels DDR3-1866"),
            ("PCIe", "40 lanes PCIe 3.0"),
        ],
    },
    "Sandy Bridge-EP": {
        "intro": [
            "Sandy Bridge-EP is the original Xeon E5-2600 series, launched in March 2012. It moved PCIe onto "
            "the processor itself and was among the first server platforms with PCIe 3.0.",
            "Sandy Bridge also introduced AVX instructions, with up to 8 cores per socket on the LGA 2011 "
            "socket.",
        ],
        "facts": [
            ("Product line", "Intel Xeon E5 (first generation)"), ("Launched", "March 2012"), ("Process", "32 nm"),
            ("Socket", "LGA 2011"), ("Max cores", "8"), ("Memory", "4 channels DDR3-1600"),
            ("PCIe", "40 lanes PCIe 3.0"),
        ],
    },
    "Westmere-EP": {
        "intro": [
            "Westmere-EP is the Xeon 5600 series, launched in March 2010. It is the 32 nm shrink of "
            "Nehalem-EP on the same LGA 1366 socket, with up to 6 cores.",
            "Westmere added AES-NI instructions, which made encryption much faster on servers.",
        ],
        "facts": [
            ("Product line", "Intel Xeon 5600 series"), ("Launched", "March 2010"), ("Process", "32 nm"),
            ("Socket", "LGA 1366"), ("Max cores", "6"), ("Memory", "3 channels DDR3"),
        ],
    },
    "Nehalem-EP": {
        "intro": [
            "Nehalem-EP is the Xeon 5500 series, launched in March 2009. It was a major redesign: the memory "
            "controller moved onto the processor, and the front-side bus was replaced by the QuickPath "
            "Interconnect.",
            "Nehalem also brought back Hyper-Threading, so its 4-core models run 8 threads.",
        ],
        "facts": [
            ("Product line", "Intel Xeon 5500 series"), ("Launched", "March 2009"), ("Process", "45 nm"),
            ("Socket", "LGA 1366"), ("Max cores", "4"), ("Memory", "3 channels DDR3"),
        ],
    },

    # ----- Others -----
    "Interlagos": {
        "intro": [
            "Interlagos is the AMD Opteron 6200 series, launched in November 2011 with the Bulldozer "
            "architecture. Each package holds two dies, for up to 16 cores on the G34 socket.",
        ],
        "facts": [
            ("Product line", "AMD Opteron 6200 series"), ("Launched", "November 2011"), ("Architecture", "Bulldozer"),
            ("Process", "32 nm"), ("Socket", "G34"), ("Max cores", "16"),
        ],
    },
    "Abu Dhabi": {
        "intro": [
            "Abu Dhabi is the AMD Opteron 6300 series, launched in November 2012 with the Piledriver "
            "architecture. It was a drop-in upgrade for Interlagos servers on the G34 socket.",
        ],
        "facts": [
            ("Product line", "AMD Opteron 6300 series"), ("Launched", "November 2012"), ("Architecture", "Piledriver"),
            ("Process", "32 nm"), ("Socket", "G34"), ("Max cores", "16"),
        ],
    },
}


# ---------- CPU families (keyed by the CSV "Family" value) ----------

FAMILIES: dict[str, str] = {
    "AMD EPYC": (
        "AMD EPYC is AMD's server processor line, introduced in 2017. Each generation is named after an "
        "Italian city (Naples, Rome, Milan, Genoa, Turin), with variants for large caches (the \"X\" models), "
        "dense cloud servers (Bergamo, Siena) and entry servers (the EPYC 4004 series)."
    ),
    "Intel Xeon 6": (
        "Intel Xeon 6 is the 2024 successor to the Xeon Scalable line. It splits into two designs: "
        "Granite Rapids with performance cores (P-cores) and Sierra Forest with efficiency cores (E-cores)."
    ),
    "Intel Xeon Platinum": (
        "Xeon Platinum is the top tier of the Intel Xeon Scalable line (2017 to 2023), with the most cores, "
        "the largest caches and support for the most sockets per server."
    ),
    "Intel Xeon Gold": (
        "Xeon Gold is the mainstream tier of the Intel Xeon Scalable line (2017 to 2023). The 5000-series "
        "models target cost-sensitive servers; the 6000-series models offer higher clocks and more memory "
        "and socket scalability."
    ),
    "Intel Xeon Silver": (
        "Xeon Silver is the entry-to-mid tier of the Intel Xeon Scalable line (2017 to 2023), used in "
        "storage, networking and general-purpose servers where cost and power matter more than peak speed."
    ),
    "Intel Xeon Bronze": (
        "Xeon Bronze is the entry tier of the Intel Xeon Scalable line, with low core counts and clocks "
        "for basic servers."
    ),
    "Intel Xeon Max": (
        "Intel Xeon Max processors are Sapphire Rapids chips with HBM2e memory inside the package, built "
        "for memory-bandwidth-bound HPC workloads."
    ),
    "Intel Xeon E5/E3": (
        "Before Xeon Scalable, Intel's main server lines were the Xeon E5 (two- and four-socket servers) and "
        "the Xeon E3 (single-socket entry servers), from Sandy Bridge in 2012 to Broadwell in 2016."
    ),
    "AMD Opteron": (
        "AMD Opteron was AMD's server processor line before EPYC, from 2003 until the mid-2010s."
    ),
}


# ---------- GPU vendors (keyed by the CSV "Vendor" value) ----------

GPU_VENDORS: dict[str, str] = {
    "NVIDIA": (
        "NVIDIA's datacenter GPUs range from inference cards such as the T4 and L4 to the Hopper (H100, "
        "H200) and Blackwell (B200, B300) accelerators used for AI training. The top models come in SXM "
        "modules for HGX servers, and some as \"Superchips\" that pair GPUs with NVIDIA's Grace CPU."
    ),
    "AMD": (
        "AMD Instinct is AMD's line of datacenter accelerators for HPC and AI. Recent models such as the "
        "MI300X, MI325X and MI355X use the OAM module format and very large HBM memory, while the MI300A "
        "combines CPU and GPU cores in one package."
    ),
    "Intel": (
        "Intel's Data Center GPU Max series (Ponte Vecchio) is a multi-tile accelerator for HPC, used in "
        "the Aurora supercomputer."
    ),
}


# ---------- Overview pages ----------

CPUS_INDEX_GUIDE = [
    ("Cores and threads",
     "Cores are independent processing units. With simultaneous multithreading (Hyper-Threading on "
     "Intel, SMT on AMD) each core runs two threads; some models, such as Intel's E-core Xeons, run one."),
    ("Max turbo frequency",
     "The highest clock speed a single core can reach for short periods. All-core clocks under full load "
     "are lower."),
    ("L3 cache",
     "Fast memory on the processor shared between cores. Larger caches help databases, simulations and "
     "other workloads with big working sets."),
    ("TDP",
     "Thermal design power: the sustained power the cooling system must handle. Many servers let you "
     "configure a lower or higher limit."),
    ("Max memory",
     "The largest amount of RAM one processor supports, usually limited by the number of memory channels "
     "and the largest supported modules."),
]

GPUS_INDEX_GUIDE = [
    ("Memory size and type",
     "Training and serving large AI models is often limited by GPU memory. High-end accelerators use "
     "stacked HBM memory, which is much faster than the GDDR memory on PCIe inference cards."),
    ("Form factor",
     "PCIe cards fit standard servers. SXM (NVIDIA) and OAM (AMD, Intel) modules sit on a dedicated "
     "board with fast GPU-to-GPU links and higher power limits."),
    ("TDP",
     "The maximum sustained power of the GPU. Current top accelerators draw 700 W to over 1,000 W each, "
     "which usually requires liquid cooling at full density."),
]


# ---------- Hand-picked comparisons: (kind, slug A, slug B) ----------
# Only pairs people actually search for. The URL is /compare/<A>-vs-<B>;
# the reversed order redirects there. Pairs whose models are missing are skipped.

COMPARISONS: list[tuple[str, str, str]] = [
    ("cpu", "amd-epyc-9654", "intel-xeon-platinum-8480-plus"),
    ("cpu", "amd-epyc-9965", "intel-xeon-6980p"),
    ("cpu", "amd-epyc-9755", "intel-xeon-6980p"),
    ("cpu", "amd-epyc-9755", "amd-epyc-9654"),
    ("cpu", "amd-epyc-9754", "amd-epyc-9654"),
    ("cpu", "amd-epyc-9684x", "amd-epyc-9654"),
    ("cpu", "amd-epyc-9654", "amd-epyc-9554"),
    ("cpu", "intel-xeon-platinum-8592-plus", "intel-xeon-platinum-8480-plus"),
    ("cpu", "intel-xeon-6980p", "intel-xeon-platinum-8592-plus"),
    ("cpu", "intel-xeon-6780e", "amd-epyc-9754"),
    ("cpu", "amd-epyc-7763", "intel-xeon-platinum-8380"),
    ("cpu", "amd-epyc-7763", "amd-epyc-7713"),
    ("cpu", "amd-epyc-7773x", "amd-epyc-7763"),
    ("cpu", "amd-epyc-7763", "amd-epyc-7742"),
    ("cpu", "intel-xeon-platinum-8380", "intel-xeon-platinum-8280"),
    ("gpu", "nvidia-h100-sxm", "nvidia-a100-sxm-80gb"),
    ("gpu", "nvidia-h200-sxm", "nvidia-h100-sxm"),
    ("gpu", "nvidia-b200", "nvidia-h100-sxm"),
    ("gpu", "nvidia-b300", "nvidia-b200"),
    ("gpu", "amd-instinct-mi300x", "nvidia-h100-sxm"),
    ("gpu", "amd-instinct-mi325x", "nvidia-h200-sxm"),
    ("gpu", "amd-instinct-mi355x", "nvidia-b200"),
    ("gpu", "nvidia-h100-sxm", "nvidia-h100-pcie"),
    ("gpu", "nvidia-l40s", "nvidia-a100-pcie-80gb"),
    ("gpu", "nvidia-l4", "nvidia-t4"),
]
