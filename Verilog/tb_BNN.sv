
module tb_BNN ();

    // input
    logic clk;
    logic rst_n;
    logic i_valid;
    logic [783:0] i_data;
    logic [784*256-1:0] i_weight_fc1;
    logic [256*10 -1:0] i_weight_fc2;
    logic [10*256-1:0] i_threshold;

    // output
    wire [3 :0] o_result;
    wire o_valid;

    // reg
    reg [783:0] r_data [0:9999];
    reg [783:0] r_weight_fc1 [0:255];
    reg [255:0] r_weight_fc2 [0:9];
    reg [9:0] r_threshold [0:255];

    // 정답
    logic [3:0] gt_labels [0:9999];  // 최대 10000개 저장 가능
    int correct = 0;
    int total   = 10000;  // 총 테스트 개수
    string fname;

    int img_idx = 0;
    int out_idx = 0;

    int img_queue [0:12000];  // 입력 인덱스 기억용
    int head = 0;
    int tail = 0;
    int true_idx = 0;

    int logfile;

    // clk
    always #5 clk = ~clk;

    // DUT
    BNN DUT (
        .clk(clk),
        .rst_n(rst_n),
        .i_valid(i_valid),
        .i_data(i_data),
        .i_weight_fc1(i_weight_fc1),
        .i_weight_fc2(i_weight_fc2),
        .i_threshold(i_threshold),
        .o_result(o_result),
        .o_valid(o_valid)
    );

    initial begin
        //$dumpfile("dump.vcd");
        //$dumpvars(1, tb_BNN);
        logfile = $fopen("accuracy_log.txt", "w");

        clk = 0; rst_n = 0; i_valid = 0; i_data = 0; i_weight_fc1 = 0; i_weight_fc2 = 0; i_threshold = 0;

        #10;
        // data
        $readmemb("data/mnist_test_bin_images.txt", r_data);
        // weight
        $readmemb("data/fc1_weight_bin.txt", r_weight_fc1);
        $readmemb("data/fc2_weight_bin.txt", r_weight_fc2);
        $readmemb("data/fc1_threshold_bin.txt", r_threshold);

        // answer
        $readmemb("data/labels.txt", gt_labels);

        for (int i = 0; i < 256; i++) begin
            i_weight_fc1[784*i +: 784] = r_weight_fc1[i];
            i_threshold[10*i +: 10] = r_threshold[i];
        end
        for (int i = 0; i < 10; i++) begin
            i_weight_fc2[256*i +: 256] = r_weight_fc2[i];
        end

        #20;
        rst_n = 1;
        #10;

        // === 연속 파이프라인 입력 === //
        fork
            // 입력 루프
            begin
                for (img_idx = 0; img_idx < total; img_idx++) begin
                    img_queue[tail] = img_idx;
                    tail++;

                    i_data = r_data[img_idx];
                    i_valid = 1;
                    @(posedge clk);
                end
                i_valid = 0;
            end

            // 출력 루프
            begin
                wait(o_valid);
                forever begin
                    if (o_valid) begin
                        true_idx = img_queue[head+1];
                        head++;

                        if (o_result == gt_labels[true_idx])
                            correct++;

                        if ((true_idx + 1) % 1 == 0 || (true_idx + 1) == total) begin
                            $display("Time : %4t | Progress: %0d / %0d | Accuracy: %0.2f%%",
                                    $time, true_idx + 1, total, correct * 100.0 / (true_idx + 1));
                            $fdisplay(logfile, "%d %0.2f", true_idx + 1, correct * 100.0 / (true_idx + 1));
                        end

                        if (true_idx + 1 == total) begin
                            $display("Time : %4t |✅ Final Accuracy: %0.2f%%", $time, correct * 100.0 / total);
                            $fclose(logfile);
                            $finish;
                        end
                    end
                    @(posedge clk);
                end
            end
        join
    end
endmodule