# Third-party tap launcher. Python dependencies are resolved by uv on first use;
# this is deliberately not a homebrew/core Python formula.
class Sokudan < Formula
  desc "Japanese decision model with typed answers and probabilities"
  homepage "https://github.com/hiroki-abe-58/sokudan"
  url "https://files.pythonhosted.org/packages/e4/88/1a5b5b9aa4ca08a43fde22016f5d179ccc3e0c73f824230f567af3f06a44/sokudan-0.3.0.tar.gz"
  sha256 "15c3e67b56a3aebf3496ecf896a34dfc6df2c98eb1658380f794d3b074f13c0d"
  license "Apache-2.0"

  depends_on "python@3.13"
  depends_on "uv"

  def install
    libexec.install Dir["*"]
    (bin/"sokudan").write <<~SH
      #!/bin/sh
      case "$1" in
        --version) echo "#{version}"; exit 0 ;;
        ""|-h|--help)
          echo "sokudan #{version}: serve | probe-position"
          echo "First use installs Python dependencies in uv's cache."
          echo "Run: sokudan serve --help"
          exit 0 ;;
      esac
      extras=serve
      if [ "$1" = probe-position ]; then extras=serve,train,bench,torch; fi
      exec "#{Formula["uv"].opt_bin}/uv" tool run \
        --python "#{Formula["python@3.13"].opt_bin}/python3.13" \
        --from "#{libexec}[$extras]" sokudan "$@"
    SH
  end

  def caveats
    <<~EOS
      This tap installs a launcher and the Python source, not a prebuilt ML runtime.
      First use needs network access to install Python dependencies in uv's cache.
      Starting the server also downloads the Hugging Face model weights.
      Run `sokudan serve --port 8000` to start the local server.
    EOS
  end

  test do
    assert_equal version.to_s, shell_output("#{bin}/sokudan --version").strip
    assert_match "serve", shell_output("#{bin}/sokudan --help")
  end
end
