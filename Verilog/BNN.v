// ✅ Pipeline-capable BNN Top Module (partial concept)
module BNN (
    input wire clk,
    input wire rst_n,
    input wire i_valid,
    input wire [783:0] i_data,
    output reg [3:0] o_result,
    output reg o_valid
);

    reg [783:0] weight_fc1 [0:255];
    reg [255:0] weight_fc2 [0:9];
    reg [9:0] threshold [0:255];
    initial $readmemb("Verilog/data/fc1_weight_bin.txt", weight_fc1);
    initial $readmemb("Verilog/data/fc2_weight_bin.txt", weight_fc2);
    initial $readmemb("Verilog/data/fc1_threshold_bin.txt", threshold);

    // Pipeline register stage 0: Input latch
    reg [783:0] r_i_data;
    reg         r_i_valid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_i_data <= 0;
            r_i_valid <= 0;
        end else begin
            r_i_data <= i_data;
            r_i_valid <= i_valid;
        end
    end

    // FC1 output buffer for 256 outputs
    wire [255:0] fc1_result;
    wire [255:0] fc1_valid;

    genvar k;
    generate
        for (k = 0; k < 256; k = k + 1) begin
            fc_784to256 u_fc1 (
                .clk(clk),
                .rst_n(rst_n),
                .i_valid(r_i_valid),
                .i_data(r_i_data),
                .i_weight(weight_fc1[k]),
                .i_threshold(threshold[k]),
                .o_result(fc1_result[255-k]),
                .o_valid(fc1_valid[k])
            );
        end
    endgenerate

    // Delay fc1_result until all valid
    reg fc1_ready;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) fc1_ready <= 0;
        else        fc1_ready <= &fc1_valid;
    end

    // FC2 (10 classes)
    wire [99:0] fc2_result; // 10 x 10bit
    wire [9:0]  fc2_valid;
    generate
        for (k = 0; k < 10; k = k + 1) begin
            fc_256to10 u_fc2 (
                .clk(clk),
                .rst_n(rst_n),
                .i_valid(fc1_ready),
                .i_data(fc1_result),
                .i_weight(weight_fc2[k]),
                .o_result(fc2_result[10*k +: 10]),
                .o_valid(fc2_valid[k])
            );
        end
    endgenerate

    wire [3:0] argmax_result;
    wire       argmax_valid;

    argmax u_argmax (
        .clk(clk),
        .rst_n(rst_n),
        .i_valid(&fc2_valid),
        .i_data(fc2_result),
        .o_idx(argmax_result),
        .o_valid(argmax_valid)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            o_result <= 0;
            o_valid  <= 0;
        end else begin
            if (argmax_valid) begin
                o_result <= argmax_result;
                o_valid  <= 1;
            end else begin
                o_valid <= 0;
            end
        end
    end

endmodule
