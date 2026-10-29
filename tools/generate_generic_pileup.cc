// Generate a new Delphes-example minimum-bias bank with its native writer.
// Link the pinned Delphes writer/XDR sources directly; no detector simulation.
#include "Pythia8/Pythia.h"
#include "classes/DelphesPileUpWriter.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char **argv) {
  if (argc != 7) {
    std::cerr << "usage: generate_generic_pileup card output events seed xmldir settings\n";
    return 2;
  }
  try {
    const std::string output(argv[2]), partial(output + ".partial");
    if (std::ifstream(output) || std::ifstream(partial))
      throw std::runtime_error("Preserve existing output/partial");
    const int requested = std::stoi(argv[3]), seed = std::stoi(argv[4]);
    if (requested <= 0 || seed <= 0 || seed > 900000000)
      throw std::runtime_error("Invalid event count or seed");
    Pythia8::Pythia pythia(argv[5]);
    if (!pythia.readFile(argv[1]) ||
        !pythia.readString("Random:setSeed = on") ||
        !pythia.readString("Random:seed = " + std::to_string(seed)) ||
        !pythia.init()) throw std::runtime_error("Pythia initialization failed");
    if (!pythia.settings.writeFile(argv[6], true))
      throw std::runtime_error("Settings dump failed");
    auto started = std::chrono::steady_clock::now();
    long long particles = 0;
    int attempts = 0, failed = 0, accepted = 0;
    {
      DelphesPileUpWriter writer(partial.c_str());
      while (accepted < requested) {
        ++attempts;
        if (!pythia.next()) {
          if (++failed >= 3) throw std::runtime_error("Three failed generator calls");
          continue;
        }
        if (pythia.info.weight() != 1.0)
          throw std::runtime_error("Expected unit generator weight");
        for (int i = 0; i < pythia.event.size(); ++i) {
          const auto &p = pythia.event[i];
          if (!p.isFinal()) continue;
          writer.WriteParticle(p.id(), p.xProd(), p.yProd(), p.zProd(),
                               p.tProd(), p.px(), p.py(), p.pz(), p.e());
          ++particles;
        }
        writer.WriteEntry();
        if (++accepted % 1000 == 0)
          std::cout << "Accepted " << accepted << " interactions\n" << std::flush;
      }
      writer.WriteIndex();
    }
    if (std::rename(partial.c_str(), output.c_str()) != 0)
      throw std::runtime_error("Atomic output rename failed");
    double wall = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    std::ofstream report(output + ".generation.json");
    report << "{\n  \"events\": " << accepted << ",\n  \"attempts\": " << attempts
           << ",\n  \"failures\": " << failed << ",\n  \"particles\": " << particles
           << ",\n  \"seed\": " << seed << ",\n  \"pythia_version\": " << PYTHIA_VERSION
           << ",\n  \"all_generator_weights_one\": true,\n  \"generation_wall_seconds\": "
           << wall << "\n}\n";
    if (!report) throw std::runtime_error("Generation report write failed");
    pythia.stat();
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
