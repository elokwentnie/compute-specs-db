"""Shared helpers for names, slugs and CPU generations."""

import re


def display_name(model_name: str) -> str:
    """Strip trademark symbols and suffixes from a model name.

    Example:
        ``"Intel(R) Xeon(R) Gold 6330 CPU @ 2.00"`` -> ``"Intel Xeon Gold 6330"``
    """
    name = re.sub(r"\s*\((?:R|TM)\)", "", model_name or "", flags=re.IGNORECASE)
    name = re.sub(r"\s*@\s*[\d.]+\s*(?:GHz)?\s*$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\b\d+-Core\b", "", name)
    name = re.sub(r"\b(?:CPU|Processor)\b", "", name)
    return re.sub(r"\s+", " ", name).strip() or (model_name or "").strip()


def slugify(text: str) -> str:
    """Convert text to a URL slug, e.g. ``8480+`` -> ``8480-plus``."""
    text = (text or "").lower().replace("+", " plus")
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")


def model_slug(model_name: str) -> str:
    """Return the URL slug for a CPU or GPU model name."""
    return slugify(display_name(model_name))


def determine_cpu_generation(cpu_model: str, launch_year: int, family: str = None) -> str:
    """Guess the generation codename of an AMD EPYC or Intel Xeon CPU.

    Args:
        cpu_model: Model string, e.g. ``"EPYC 7301"`` or ``"Gold 6240"``.
        launch_year: Launch year.
        family: CPU family, e.g. ``"AMD EPYC"``.

    Returns:
        Codename such as ``"Rome"`` or ``"Sapphire Rapids"``, or ``""`` if unknown.
    """
    if not cpu_model or not launch_year:
        return ""
    
    cpu_model_upper = str(cpu_model).upper().strip()
    family_upper = str(family or "").upper().strip()
    
    try:
        year = int(launch_year)
        
        # AMD EPYC processors
        if "EPYC" in cpu_model_upper or "EPYC" in family_upper:
            # EPYC 7xxx series
            if "EPYC 7" in cpu_model_upper or "7" in cpu_model_upper:
                if year == 2017:
                    return "Naples"
                elif year == 2019 or year == 2020:
                    return "Rome"
                elif year == 2021 or year == 2022:
                    return "Milan"
            
            # EPYC 9xxx series (Genoa/Bergamo)
            elif "EPYC 9" in cpu_model_upper or ("9" in cpu_model_upper and "EPYC" in cpu_model_upper):
                if year == 2022 or year == 2023:
                    return "Genoa"
            
            # EPYC 8xxx series (Siena)
            elif "EPYC 8" in cpu_model_upper:
                if year >= 2023:
                    return "Siena"
            
            # EPYC 4xxx series (Genoa variants)
            elif "EPYC 4" in cpu_model_upper:
                if year == 2023:
                    return "Genoa"
        
        # Intel Xeon processors
        elif "XEON" in cpu_model_upper or "XEON" in family_upper:
            # Extract model number pattern
            # Xeon Scalable 1st Gen (Skylake): 8xxx series, launched 2017-2018
            # Xeon Scalable 2nd Gen (Cascade Lake): 6xxx series, launched 2019-2020
            # Xeon Scalable 3rd Gen (Ice Lake): 5xxx series, launched 2021
            # Xeon Scalable 4th Gen (Sapphire Rapids): 8xxx series, launched 2023
            # Xeon Scalable 5th Gen (Emerald Rapids): 8xxx series, launched 2023-2024
            
            # Check for model numbers in the format: Gold 6240, Platinum 8368, etc.
            import re
            
            # Pattern to match model numbers: 8xxx, 6xxx, 5xxx, 4xxx
            model_match = re.search(r'(\d{4})', cpu_model_upper)
            if model_match:
                model_num = int(model_match.group(1))
                first_digit = model_num // 1000
                
                # 8xxx series
                if first_digit == 8:
                    if year == 2017 or year == 2018:
                        return "Skylake"
                    elif year == 2023 or year == 2024:
                        # Distinguish between Sapphire Rapids and Emerald Rapids
                        # Sapphire Rapids: 83xx, 84xx, 85xx
                        # Emerald Rapids: 85xx (some), 86xx, 87xx
                        if model_num >= 8500 and model_num < 8600:
                            # Could be either, default to Sapphire Rapids for 2023
                            return "Sapphire Rapids" if year == 2023 else "Emerald Rapids"
                        elif model_num >= 8600:
                            return "Emerald Rapids"
                        else:
                            return "Sapphire Rapids"
                
                # 6xxx series (Cascade Lake)
                elif first_digit == 6:
                    if year == 2019 or year == 2020:
                        return "Cascade Lake"
                
                # 5xxx series (Ice Lake)
                elif first_digit == 5:
                    if year == 2021:
                        return "Ice Lake"
                
                # 4xxx series (Ice Lake - some variants)
                elif first_digit == 4:
                    if year == 2021:
                        return "Ice Lake"
            
            # Fallback: determine by year for Xeon Scalable
            if "SCALABLE" in family_upper or "GOLD" in family_upper or "PLATINUM" in family_upper or "SILVER" in family_upper:
                if year == 2017 or year == 2018:
                    return "Skylake"
                elif year == 2019 or year == 2020:
                    return "Cascade Lake"
                elif year == 2021:
                    return "Ice Lake"
                elif year == 2023:
                    return "Sapphire Rapids"
                elif year == 2024:
                    return "Emerald Rapids"
        
        # Legacy Intel Xeon (E5, E3, etc.)
        elif "XEON" in family_upper and ("E5" in cpu_model_upper or "E3" in cpu_model_upper):
            # E5 v2 = Ivy Bridge (2013)
            # E5 v3 = Haswell (2014)
            # E5 v4 = Broadwell (2016)
            if "V2" in cpu_model_upper or "V 2" in cpu_model_upper:
                if year == 2013:
                    return "Ivy Bridge"
            elif "V3" in cpu_model_upper or "V 3" in cpu_model_upper:
                if year == 2014:
                    return "Haswell"
            elif "V4" in cpu_model_upper or "V 4" in cpu_model_upper:
                if year == 2016:
                    return "Broadwell"
    
    except (ValueError, TypeError):
        pass
    
    return ""


# Alias for backward compatibility
def determine_epyc_generation(cpu_model: str, launch_year: int, family: str = None) -> str:
    """Deprecated alias of ``determine_cpu_generation``."""
    return determine_cpu_generation(cpu_model, launch_year, family)

